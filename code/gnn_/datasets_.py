import pandas as pd
from pathlib import Path
from rdkit import Chem
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, InMemoryDataset, Data
from torch_geometric.transforms import Compose as g_Compose
from tqdm import tqdm
from typing import List, Dict, Tuple

from data_proc.mol_preproc import create_processor_list


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

class HIVMoleculeDataset(InMemoryDataset):
    def __init__(
        self, root: str|Path, filename: str, transform=None,
        pre_transform=None, size_increase: float|None=None
    ) -> None:
        self.filename = filename

        self.atom_to_idx = {}
        self.bond_to_idx = {}
        self.n_atom_types = 0
        self.n_bond_types = 0
        self.max_n_atoms = 0
        self.size_increase = size_increase

        super().__init__(
            root, transform=transform, pre_transform=pre_transform)
        
        self.load(self.processed_paths[0])
        metadata = torch.load(self.processed_paths[1], weights_only=False)
        self.atom_to_idx = metadata["atom_to_idx"]
        self.bond_to_idx = metadata["bond_to_idx"]
        self.n_atom_types = metadata["n_atom_types"]
        self.n_bond_types = metadata["n_bond_types"]
        self.max_n_atoms = metadata["max_n_atoms"]

        self.atom_vocab = self.define_atom_vocab(self.atom_to_idx)
        self.bond_vocab = list(self.bond_to_idx.keys())

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
            Chem.Kekulize(mol, clearAromaticFlags=True)

            x = self._node_features(mol)
            edge_attr, edge_index = self._edge_features_list(mol)

            raw_graphs.append(Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor(row["HIV_active"], dtype=torch.long)
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
            raw_graphs, total=len(raw_graphs), desc="one-hot encode"):
            graph.x = F.one_hot(graph.x, num_classes=self.n_atom_types).float()
            graph.edge_attr = F.one_hot(
                graph.edge_attr, num_classes=self.n_bond_types).float()

            proc_graphs.append(processors(graph))

        self.save(proc_graphs, self.processed_paths[0])
        torch.save({
            "atom_to_idx": self.atom_to_idx,
            "bond_to_idx": self.bond_to_idx,
            "n_atom_types": self.n_atom_types,
            "n_bond_types": self.n_bond_types,
            "max_n_atoms": self.max_n_atoms
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