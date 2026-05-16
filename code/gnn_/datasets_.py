import math
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, InMemoryDataset, Data
from torch_geometric.transforms import Compose as g_Compose
from tqdm import tqdm
from typing import List, Dict, Tuple

from data_proc.mol_preproc import (
    TU_MUTAG_CONFIG, create_processor_list, mol_from_graph)


def _log_conditioning(
    values: Tuple[int|float|torch.Tensor, ...],
    max_values: Tuple[int|float, ...], device: torch.device|str|None=None
) -> torch.Tensor:
    vals = torch.stack([
        torch.as_tensor(v, dtype=torch.float, device=device)
        for v in values
    ])
    denoms = torch.tensor(
        [math.log1p(max(1, int(m))) for m in max_values],
        dtype=torch.float,
        device=vals.device,
    )
    return (torch.log1p(vals.clamp_min(0)) / denoms).view(1, -1)

def _maybe_repeat(
    conditioning: torch.Tensor, repeat: int|None=None
) -> torch.Tensor:
    if repeat is None:
        return conditioning
    return conditioning.repeat(repeat, 1)

def _capacity_for_n_atoms(
    n_atoms: int, size_increase: float|None, max_n_atoms: int,
    max_capacity: int=0, clamp: bool=True
) -> int:
    if size_increase is not None:
        capacity = math.ceil(n_atoms * size_increase)
    else:
        capacity = max_n_atoms

    capacity = max(n_atoms, capacity)
    if clamp and max_capacity:
        capacity = min(capacity, max_capacity)
    return capacity

class ProcessedInMemoryDataset(InMemoryDataset):
    def __init__(
        self, base_dataset: Dataset, processors=None, clone: bool=True
    ) -> None:
        self.base_dataset = base_dataset
        self.processors = g_Compose(processors or [])
        self.clone = clone

        super().__init__(root=None, transform=None)

        data_list = []
        for data in self.base_dataset:
            if self.clone:
                data = data.clone()

            data = self.processors(data)
            data_list.append(data)

        self._data, self.slices = self.collate(data_list)
    
    @property
    def data(self):
        return self._data

class MUTAGMoleculeDataset(InMemoryDataset):
    def __init__(
        self, base_dataset: Dataset, root: str|Path, transform=None,
        pre_transform=None, size_increase: float|None=None,
        max_n_atoms: int|None=None, force_reload: bool=False,
    ) -> None:
        self.base_dataset = base_dataset
        self.size_increase = size_increase
        self.max_n_atoms = max_n_atoms or TU_MUTAG_CONFIG["max_n_atoms"]

        self.atom_vocab = TU_MUTAG_CONFIG["node_list"]
        self.edge_name_vocab = list(TU_MUTAG_CONFIG["edge_dict"].keys())
        self.edge_type_vocab = list(TU_MUTAG_CONFIG["edge_dict"].values())
        self.aromatic_idx = TU_MUTAG_CONFIG["aromatic_idx"]

        self.kek_edge_name_vocab = [
            n for n in self.edge_name_vocab if n != "aromatic"
        ]
        self.kek_edge_type_vocab = [
            b for n, b in TU_MUTAG_CONFIG["edge_dict"].items()
            if n != "aromatic"
        ]

        self.max_real_n_atoms = 0
        self.max_n_heteroatoms = 0
        self.max_n_rings = 0
        self.max_capacity = 0

        super().__init__(
            root, transform=transform, pre_transform=pre_transform,
            force_reload=force_reload
        )

        self.load(self.processed_paths[0])
        metadata = torch.load(self.processed_paths[1], weights_only=False)
        self.__dict__.update(metadata)
        self.bond_token_names = getattr(
            self, "bond_token_names", self.kek_edge_name_vocab + ["no_bond"])
        if len(getattr(self, "conditioning_names", [])) != 3:
            self.conditioning_names = [
                "log_real_n_atoms",
                "log_n_heteroatoms",
                "log_n_rings",
            ]

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt", "metadata.pt"]

    def process(self) -> None:
        raw_graphs = []

        for graph in tqdm(
            self.base_dataset, total=len(self.base_dataset),
            desc="reading MUTAG"
        ):
            graph = graph.cpu().clone()

            cond_raw = self._conditioning(graph)
            n_atoms, n_heteroatoms, n_rings = cond_raw.tolist()

            self.max_real_n_atoms = max(self.max_real_n_atoms, int(n_atoms))
            self.max_n_heteroatoms = max(
                self.max_n_heteroatoms, int(n_heteroatoms))
            self.max_n_rings = max(self.max_n_rings, int(n_rings))

            graph.conditioning_raw = cond_raw.view(1, -1)
            raw_graphs.append(graph)

        trafos = create_processor_list(
            self.atom_vocab, self.edge_name_vocab, self.edge_type_vocab,
            self.aromatic_idx, max_n_atoms=self.max_n_atoms,
            size_increase=self.size_increase,
            processors=["kekulize", "pad_none", "pad_max", "to_int"],
        )
        processors = g_Compose(trafos)

        proc_graphs = []
        for graph in tqdm(
            raw_graphs, total=len(raw_graphs), desc="processing MUTAG"):
            graph = processors(graph)
            self.max_capacity = max(self.max_capacity, graph.num_nodes)

            n_atoms, n_heteroatoms, n_rings = graph.conditioning_raw.flatten()
            graph.conditioning = self.encode_conditioning(
                n_atoms, n_heteroatoms, n_rings)

            proc_graphs.append(graph)

        self.save(proc_graphs, self.processed_paths[0])
        torch.save(self._metadata(), self.processed_paths[1])

    def _conditioning(self, graph: Data) -> torch.Tensor:
        atom_ids = graph.x.argmax(dim=-1).tolist()
        symbols = [self.atom_vocab[i] for i in atom_ids]

        n_atoms = len(symbols)
        n_heteroatoms = sum(sym not in ("C", "H") for sym in symbols)
        n_rings = self._num_rings(graph)

        return torch.tensor(
            [n_atoms, n_heteroatoms, n_rings], dtype=torch.float)

    def encode_conditioning(
        self, n_atoms: int|float|torch.Tensor,
        n_heteroatoms: int|float|torch.Tensor, n_rings: int|float|torch.Tensor,
        device: torch.device|str|None=None,
    ) -> torch.Tensor:
        return _log_conditioning(
            (n_atoms, n_heteroatoms, n_rings),
            (self.max_real_n_atoms, self.max_n_heteroatoms, self.max_n_rings),
            device=device,
        )

    def sample_conditioning(
        self, n_atoms: int, n_heteroatoms: int, n_rings: int,
        repeat: int|None=None, device: torch.device|str|None=None,
    ) -> torch.Tensor:
        return _maybe_repeat(
            self.encode_conditioning(
                n_atoms, n_heteroatoms, n_rings, device=device),
            repeat=repeat,
        )

    def capacity_for_n_atoms(self, n_atoms: int, clamp: bool=True) -> int:
        return _capacity_for_n_atoms(
            n_atoms, self.size_increase, self.max_n_atoms, self.max_capacity,
            clamp=clamp,
        )

    def _num_rings(self, graph: Data) -> int:
        try:
            mol = mol_from_graph(graph, self.atom_vocab, self.edge_type_vocab)
            mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(mol)
            return int(mol.GetRingInfo().NumRings())
        except Exception:
            return self._cycle_rank(graph)

    @staticmethod
    def _cycle_rank(graph: Data) -> int:
        edges = {
            tuple(sorted((int(u), int(v))))
            for u, v in graph.edge_index.t().tolist()
            if int(u) != int(v)
        }
        return max(0, len(edges) - graph.num_nodes + 1)

    def _metadata(self) -> Dict:
        return {
            "atom_vocab": self.atom_vocab,
            "bond_vocab": self.kek_edge_type_vocab,
            "edge_name_vocab": self.edge_name_vocab,
            "edge_type_vocab": self.edge_type_vocab,
            "kek_edge_name_vocab": self.kek_edge_name_vocab,
            "kek_edge_type_vocab": self.kek_edge_type_vocab,
            "n_atom_tokens": len(self.atom_vocab) + 1,
            "n_bond_tokens": len(self.kek_edge_type_vocab) + 1,
            "max_n_atoms": self.max_n_atoms,
            "max_real_n_atoms": self.max_real_n_atoms,
            "max_n_heteroatoms": self.max_n_heteroatoms,
            "max_n_rings": self.max_n_rings,
            "max_capacity": self.max_capacity,
            "size_increase": self.size_increase,
            "bond_token_names": self.kek_edge_name_vocab + ["no_bond"],
            "conditioning_names": [
                "log_real_n_atoms",
                "log_n_heteroatoms",
                "log_n_rings",
            ],
        }

    def download(self):
        pass

class HIVMoleculeDataset(InMemoryDataset):
    def __init__(
        self, root: str|Path, filename: str, transform=None,
        pre_transform=None, size_increase: float|None=None,
        force_reload: bool=False,
    ) -> None:
        self.filename = filename

        self.atom_to_idx = {}
        self.bond_to_idx = {}
        self.n_atom_types = 0
        self.n_bond_types = 0
        self.n_atom_tokens = 0
        self.n_bond_tokens = 0
        self.max_n_atoms = 0
        self.max_n_heteroatoms = 0
        self.max_n_rings = 0
        self.max_capacity = 0
        self.size_increase = size_increase

        super().__init__(
            root, transform=transform, pre_transform=pre_transform,
            force_reload=force_reload,
        )
        
        self.load(self.processed_paths[0])
        metadata = torch.load(self.processed_paths[1], weights_only=False)
        self.atom_to_idx = metadata["atom_to_idx"]
        self.bond_to_idx = metadata["bond_to_idx"]
        self.n_atom_types = metadata["n_atom_types"]
        self.n_bond_types = metadata["n_bond_types"]
        self.n_atom_tokens = metadata.get("n_atom_tokens", self.n_atom_types + 1)
        self.n_bond_tokens = metadata.get("n_bond_tokens", self.n_bond_types + 1)
        self.max_n_atoms = metadata["max_n_atoms"]
        self.max_n_heteroatoms = metadata.get(
            "max_n_heteroatoms", self.max_n_atoms)
        self.max_n_rings = metadata["max_n_rings"]
        self.max_capacity = metadata.get(
            "max_capacity", self._infer_max_capacity())
        self.size_increase = metadata.get("size_increase", self.size_increase)

        self.atom_vocab = metadata.get(
            "atom_vocab", self.define_atom_vocab(self.atom_to_idx))
        self.bond_vocab = metadata.get("bond_vocab", list(self.bond_to_idx.keys()))
        self.bond_token_names = metadata.get(
            "bond_token_names", self.define_bond_token_names(self.bond_vocab))
        self.conditioning_names = metadata.get(
            "conditioning_names",
            ["log_real_n_atoms", "log_n_heteroatoms", "log_n_rings"],
        )

    @property
    def raw_file_names(self) -> str:
        return self.filename
    
    @property
    def processed_file_names(self) -> List[str]:
         return ["data.pt", "metadata.pt"]

    def process(self) -> None:
        df = pd.read_csv(self.raw_paths[0])

        raw_graphs = []
        for i, row in tqdm(
            df.iterrows(), total=df.shape[0], desc="parsing molecules"):
            mol = Chem.MolFromSmiles(row["smiles"])
            mol = Chem.DeleteSubstructs(mol, Chem.MolFromSmarts("[#1X0]"))
            Chem.SanitizeMol(mol)
            conditioning = self._conditioning(mol)
            Chem.Kekulize(mol, clearAromaticFlags=True)

            x = self._node_features(mol)
            edge_attr, edge_index = self._edge_features_list(mol)

            raw_graphs.append(Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor(row["HIV_active"], dtype=torch.long),
                conditioning_raw=conditioning.view(1, -1),
            ))
        
        atom_vocab = self.define_atom_vocab(self.atom_to_idx)
        trafos = create_processor_list(
            atom_vocab, list(self.bond_to_idx.values()),
            list(self.bond_to_idx.keys()), 0, max_n_atoms=self.max_n_atoms,
            size_increase=self.size_increase,
            processors=["pad_none", "pad_max", "to_int"]
        )
        processors = g_Compose(trafos)
        
        proc_graphs = []
        for graph in tqdm(
            raw_graphs, total=len(raw_graphs), desc="formatting graph features"):
            graph.x = F.one_hot(graph.x, num_classes=self.n_atom_types).float()
            graph.edge_attr = F.one_hot(
                graph.edge_attr, num_classes=self.n_bond_types).float()
            
            n_atoms, n_hetatoms, n_rings = graph.conditioning_raw.flatten()
            graph.conditioning = self.encode_conditioning(
                n_atoms, n_hetatoms, n_rings)

            graph = processors(graph)
            self.max_capacity = max(self.max_capacity, graph.num_nodes)
            proc_graphs.append(graph)

        self.save(proc_graphs, self.processed_paths[0])
        torch.save({
            "atom_to_idx": self.atom_to_idx,
            "bond_to_idx": self.bond_to_idx,
            "atom_vocab": self.define_atom_vocab(self.atom_to_idx),
            "bond_vocab": list(self.bond_to_idx.keys()),
            "bond_token_names": self.define_bond_token_names(
                list(self.bond_to_idx.keys())),
            "n_atom_types": self.n_atom_types,
            "n_bond_types": self.n_bond_types,
            "n_atom_tokens": self.n_atom_types + 1,
            "n_bond_tokens": self.n_bond_types + 1,
            "max_n_atoms": self.max_n_atoms,
            "max_n_heteroatoms": self.max_n_heteroatoms,
            "max_n_rings": self.max_n_rings,
            "max_capacity": self.max_capacity,
            "size_increase": self.size_increase,
            "conditioning_names": [
                "log_real_n_atoms",
                "log_n_heteroatoms",
                "log_n_rings",
            ],
        }, self.processed_paths[1])

    def _node_features(self, mol) -> torch.Tensor:
        feat_list = [
            self._token_id(self.atom_to_idx, atom.GetAtomicNum())
            for atom in mol.GetAtoms()
        ]
        self.n_atom_types = len(self.atom_to_idx)
        n_atoms = len(feat_list)
        if n_atoms > self.max_n_atoms:
            self.max_n_atoms = n_atoms
        return torch.tensor(feat_list, dtype=torch.long)
    
    def _edge_features_list(self, mol) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_list = []
        edge_list = []
        for bond in mol.GetBonds():
            feat = self._token_id(self.bond_to_idx, bond.GetBondType())
            feat_list.extend([feat, feat])

            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_list.extend([[i, j], [j, i]])

        self.n_bond_types = len(self.bond_to_idx)
        if len(edge_list) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return torch.tensor(feat_list, dtype=torch.long), edge_index

    def _conditioning(self, mol) -> torch.Tensor:
        n_atoms = mol.GetNumAtoms()
        n_heteroatoms = rdMolDescriptors.CalcNumHeteroatoms(mol)
        n_rings = rdMolDescriptors.CalcNumRings(mol)
        if n_heteroatoms > self.max_n_heteroatoms:
            self.max_n_heteroatoms = n_heteroatoms
        if n_rings > self.max_n_rings:
            self.max_n_rings = n_rings
        return torch.tensor(
            [n_atoms, n_heteroatoms, n_rings], dtype=torch.float)

    def encode_conditioning(
        self, n_atoms: int|float|torch.Tensor,
        n_heteroatoms: int|float|torch.Tensor, n_rings: int|float|torch.Tensor,
        device: torch.device|str|None=None,
    ) -> torch.Tensor:
        return _log_conditioning(
            (n_atoms, n_heteroatoms, n_rings),
            (self.max_n_atoms, self.max_n_heteroatoms, self.max_n_rings),
            device=device,
        )

    def sample_conditioning(
        self, n_atoms: int, n_heteroatoms: int, n_rings: int,
        repeat: int|None=None, device: torch.device|str|None=None,
    ) -> torch.Tensor:
        return _maybe_repeat(
            self.encode_conditioning(
                n_atoms, n_heteroatoms, n_rings, device=device),
            repeat=repeat,
        )

    def capacity_for_n_atoms(self, n_atoms: int, clamp: bool=True) -> int:
        return _capacity_for_n_atoms(
            n_atoms, self.size_increase, self.max_n_atoms, self.max_capacity,
            clamp=clamp,
        )

    def _token_id(self, vocab: Dict, token: int|float) -> Dict:
        if token not in vocab:
            vocab[token] = len(vocab)
        return vocab[token]
    
    def download(self):
        pass

    @staticmethod
    def define_atom_vocab(idx_dict: Dict[int, int]) -> List[str]:
        pt = Chem.GetPeriodicTable()
        return [pt.GetElementSymbol(a_num) for a_num in idx_dict.keys()]

    @staticmethod
    def define_bond_token_names(
        bond_vocab: List[Chem.rdchem.BondType],
    ) -> List[str]:
        name_map = {
            Chem.rdchem.BondType.SINGLE: "single",
            Chem.rdchem.BondType.DOUBLE: "double",
            Chem.rdchem.BondType.TRIPLE: "triple",
            Chem.rdchem.BondType.AROMATIC: "aromatic",
        }
        names = [
            name_map.get(bond_type, f"bond_{int(bond_type)}")
            for bond_type in bond_vocab
        ]
        return names + ["no_bond"]

    def _infer_max_capacity(self) -> int:
        if hasattr(self, "slices") and self.slices and "x" in self.slices:
            n_nodes = self.slices["x"][1:] - self.slices["x"][:-1]
            return int(n_nodes.max())
        return self.max_n_atoms
