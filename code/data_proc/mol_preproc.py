import numpy as np
from rdkit import Chem
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import List, Callable


TU_MUTAG_CONFIG = {
    "node_list": ["C", "N", "O", "F", "I", "Cl", "Br"],
    "edge_dict": {
        "aromatic": Chem.rdchem.BondType.AROMATIC,
        "single": Chem.rdchem.BondType.SINGLE,
        "double": Chem.rdchem.BondType.DOUBLE,
        "triple": Chem.rdchem.BondType.TRIPLE
    },
    "aromatic_idx": 0,
    "max_n_atoms": 35
}

def create_processor_list(
    atom_name_vocab: List[str], bond_name_vocab: List[str],
    bond_type_vocab: List[Chem.rdchem.BondType], aromatic_idx: int,
    max_n_atoms: int|None=None, size_increase: float|None=None,
    processors: List[str]=["kekulize", "pad_none", "pad_max", "to_int"]
) -> List[Callable]:
    trafos = []
    if "kekulize" in processors:
        kek_wrapper = lambda dat: kekulize_graph(
            dat, atom_name_vocab, bond_name_vocab, bond_type_vocab,
            aromatic_idx
        )
        trafos.append(kek_wrapper)
    if "pad_none" in processors:
        trafos.append(pad_graph_w_none_tokens)
    if "pad_max" in processors:
        pad_wrapper = lambda dat: pad_nodes_and_edges_to_max(
            dat, max_n_atoms=max_n_atoms, size_increase=size_increase)
        trafos.append(pad_wrapper)
    if "to_int" in processors:
        trafos.append(convert_to_int)
    return trafos

def mol_from_graph(
    data: Data, atom_name_vocab: List[str],
    bond_type_vocab: List[Chem.rdchem.BondType]
):
    mol_i = Chem.RWMol()
    node_to_idx = {}
    for i, v in enumerate(data.x):
        a = Chem.Atom(atom_name_vocab[np.argmax(v.cpu())])
        mol_idx = mol_i.AddAtom(a)
        node_to_idx[i] = mol_idx

    edges_i = []
    for i, e in enumerate(data.edge_index.T):
        e = e.cpu().numpy().tolist()

        if e[::-1] not in edges_i:
            edges_i.append(e)
        else:
            continue

        edge_feat = data.edge_attr[i]
        bond_type = bond_type_vocab[np.argmax(edge_feat.cpu())]
        mol_i.AddBond(node_to_idx[e[0]], node_to_idx[e[1]], bond_type)

    mol = mol_i.GetMol()
    return mol

def kekulize_graph(
    data: Data, atom_name_vocab: List[str], bond_name_vocab: List[str],
    bond_type_vocab: List[Chem.rdchem.BondType], aromatic_idx: int
) -> Data:
    mol = mol_from_graph(data, atom_name_vocab, bond_type_vocab)
    Chem.Kekulize(mol, clearAromaticFlags=True)

    bond_map = {}
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_map[(u, v)] = bond.GetBondType()
        bond_map[(v, u)] = bond.GetBondType()

    new_edge_attrs = []
    for u, v in data.edge_index.t().tolist():
        bt = bond_map.get((u, v), Chem.rdchem.BondType.SINGLE)
        bidx = bond_type_vocab.index(bt)
        one_hot = [1 if i == bidx else 0 for i in range(len(bond_name_vocab))]
        new_edge_attrs.append(one_hot)
    edge_attrs = torch.tensor(new_edge_attrs, dtype=torch.float)
    # remove the edge_feature column associated with the aromatic bond type
    # as it has been removed through kekulization
    edge_attrs = edge_attrs[:, [
        c_idx for c_idx in range(len(bond_name_vocab)) if c_idx != aromatic_idx
    ]]

    return Data(
        x=data.x.clone(),
        edge_index=data.edge_index.clone(),
        edge_attr=edge_attrs,
        **{
            k: v
            for k, v in data.to_dict().items()
            if k not in ("x", "edge_index", "edge_attr")
        }
    )

def fix_nitro_charges(mol):
    mol = Chem.RWMol(mol)
    mol.UpdatePropertyCache(strict=False)

    patt = Chem.MolFromSmarts("[N;+0;D3](=[O;+0])-[O;+0]")

    for n, o_dbl, o_sng in mol.GetMol().GetSubstructMatches(patt):
        mol.GetAtomWithIdx(n).SetFormalCharge(+1)
        mol.GetAtomWithIdx(o_sng).SetFormalCharge(-1)

    out = mol.GetMol()
    out.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(out)
    return out

def pad_graph_w_none_tokens(data: Data) -> Data:
    x = data.x.clone()
    x = F.pad(x, (0, 1), mode="constant", value=0)
    edge_attrs = data.edge_attr.clone()
    edge_attrs = F.pad(edge_attrs, (0, 1), mode="constant", value=0)

    return Data(
        x=x,
        edge_index=data.edge_index.clone(),
        edge_attr=edge_attrs,
        **{
            k: v
            for k, v in data.to_dict().items()
            if k not in ("x", "edge_index", "edge_attr")
        }
    )

def pad_nodes_and_edges_to_max(
    data: Data, max_n_atoms: int|None=None, size_increase: float|None=None
) -> Data:
    assert max_n_atoms is not None or size_increase is not None, (
        "Either of max_n_atoms or size_increase has to be specified!")
    data = data.clone()
    if size_increase:
        import math
        max_n_atoms = math.ceil(data.num_nodes * size_increase)
    
    # 1. Pad Nodes to "no-atom"-Type
    node_attr = data.x
    n_node_tokens = node_attr.size(-1)
    non_atom_row = torch.tensor(
        [0]*(n_node_tokens - 1) + [1], dtype=node_attr.dtype,
        device=node_attr.device
    ).unsqueeze(0)
    padded_node_attr = torch.cat((
        node_attr, non_atom_row.expand(max_n_atoms - node_attr.shape[0], -1)
    ), dim=0)
    data.x = padded_node_attr

    # 2. Pad Edges to "no-bond"-Type
    edge_index, edge_attr = data.edge_index, data.edge_attr
    n_edge_tokens = edge_attr.size(-1)

    no_bond_row = torch.tensor(
        [0]*(n_edge_tokens - 1) + [1], dtype=edge_attr.dtype,
        device=edge_attr.device
    ).unsqueeze(0)

    row, col = torch.triu_indices(
        max_n_atoms, max_n_atoms, offset=1, device=edge_index.device)
    padded_edge_index = torch.stack([row, col], dim=0)
    n_padded_edges = padded_edge_index.size(1)

    padded_edge_attr = no_bond_row.expand(n_padded_edges, -1).clone()
    pair_to_idx = {
        (int(u), int(v)): i
        for i, (u, v) in enumerate(padded_edge_index.t().tolist())
    }
    for edge_idx, (u, v) in enumerate(edge_index.t().tolist()):
        if u == v:
            continue

        i, j = sorted((u, v))
        new_idx = pair_to_idx[(i, j)]
        padded_edge_attr[new_idx] = edge_attr[edge_idx]
    

    data.edge_index = torch.cat(
        (padded_edge_index, padded_edge_index.flip(0)), dim=1)
    data.edge_attr = torch.cat((padded_edge_attr, padded_edge_attr), dim=0)

    return data

def convert_to_int(data: Data) -> Data:
    batch = data.clone()

    batch.x = batch.x.long()
    batch.edge_attr = batch.edge_attr.long()

    return batch

def clean_up_graph(data: Data, max_n_atoms: int|None=None) -> Data:
    if max_n_atoms is None:
        max_n_atoms = data.num_nodes

    x = data.x.clone()
    edge_index = data.edge_index.clone()
    edge_attr = data.edge_attr.clone()

    # get a mask filtering out the "no-node"-type and the associated indices
    real_x_mask = (x[:, -1] == 0)
    real_x_idx = torch.nonzero(real_x_mask, as_tuple=False).flatten()

    # overwrite the node features
    x = x[real_x_idx][:, :-1]

    # construct a mapping of src,tgt nodes from their old to their new index
    # based on the new updated node indices
    node_remapping = torch.full(
        (max_n_atoms,), -1, dtype=torch.long, device=x.device)
    node_remapping[real_x_idx] = torch.arange(
        real_x_idx.size(0), device=x.device)

    # get old source and target node indices
    src, tgt = edge_index
    # also get mask for "no-bond"-type
    real_edge_mask = edge_attr[:, -1] == 0
    edges_to_keep = real_x_mask[src] & real_x_mask[tgt] & real_edge_mask
    # filter out removed indices
    src_filt = src[edges_to_keep]
    tgt_filt = tgt[edges_to_keep]
    # map old indices to new ones
    src_new = node_remapping[src_filt]
    tgt_new = node_remapping[tgt_filt]

    # overwrite edge index and edge attributes
    edge_index = torch.stack((src_new, tgt_new), dim=0)
    edge_attr = edge_attr[edges_to_keep][:, :-1]

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        **{
            k: v
            for k, v in data.__dict__.items()
            if k not in ("x", "edge_index", "edge_attr")
        }
    )

def batch_to_mols(
    batch: Data, atom_vocab: List[str],
    bond_type_vocab: List[Chem.rdchem.BondType], max_n_atoms: int|None=None
) -> List:
    indv_graphs = []
    edge_graph = batch.batch[batch.edge_index[0]]
    for graph_idx in range(batch.num_graphs):
        node_mask = batch.batch == graph_idx
        edge_mask = edge_graph == graph_idx

        x = batch.x[node_mask]
        edge_index = batch.edge_index[:, edge_mask]
        edge_attr = batch.edge_attr[edge_mask]

        g = Data(
            x=x, edge_index=edge_index - edge_index.min(),
            edge_attr=edge_attr, y=batch.y[graph_idx]
        )
        indv_graphs.append(g)
    
    indv_mols = []
    for g in indv_graphs:
        g_clean = clean_up_graph(g, max_n_atoms)
        mol = mol_from_graph(g_clean, atom_vocab, bond_type_vocab)
        indv_mols.append(mol)
    
    return indv_mols
