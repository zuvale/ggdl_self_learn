import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from torch_geometric.data import InMemoryDataset, Data, Batch
from typing import Tuple


def graph_initial_samples(
    base_dist: nn.Module, batch_size: int, max_n_nodes: int,
    y: torch.Tensor|None=None, device: str="cpu",
    conditioning: torch.Tensor|None=None
) -> Data:
    data_list = []

    row, col = torch.triu_indices(
        max_n_nodes, max_n_nodes, offset=1, device=device)
    triu_edge_index = torch.stack([row, col], dim=0)
    num_edges = triu_edge_index.size(1)
    edge_index = torch.cat(
        [triu_edge_index, triu_edge_index.flip(0)], dim=1)

    node_samples, upper_edge_samples = base_dist.sample(
        node_sample_shape=(batch_size, max_n_nodes),
        edge_sample_shape=(batch_size, num_edges),
    )

    for i in range(batch_size):
        x, upper_edge_attr = node_samples[i], upper_edge_samples[i]
        
        edge_attr = torch.cat([upper_edge_attr, upper_edge_attr], dim=0)
        data = Data(
            x=x,
            edge_index=edge_index.clone(),
            edge_attr=edge_attr,
            y=y[i].view(1) if y is not None else None,
            num_nodes=max_n_nodes,
        )
        if conditioning is not None:
            data.conditioning = conditioning[i].view(1, -1)

        data_list.append(data)

    return Batch.from_data_list(data_list).to(device)

class GaussianBase(nn.Module):
    normalization_constant = 0.5 * torch.log(
        torch.Tensor([2 * torch.pi]))
    def __init__(self, dimensionality: Tuple[int]) -> None:
        super().__init__()

        self.dim = dimensionality
        self.mean = nn.Parameter(torch.zeros(*self.dim), requires_grad=False)
        self.std = nn.Parameter(torch.ones(*self.dim), requires_grad=False)
    
    def forward(self) -> td.Distribution:
        return td.Independent(td.Normal(self.mean, self.std), len(self.dim))
    
class MixtureOfGaussiansBase(nn.Module):
    def __init__(
        self, dimensionality: Tuple[int], num_components: int,
        logit_init: float=3.0, logvar_init: float=-4.6,
        logvar_bounds: Tuple[float, float]|None=None
    ) -> None:
        super().__init__()

        self.dim = dimensionality
        self.k = num_components
        self.mix_logits = nn.Parameter(
            torch.ones(self.k) * logit_init, requires_grad=True)
        self.means = nn.Parameter(
            torch.randn(self.k, *self.dim), requires_grad=True)
        self.logvars = nn.Parameter(
            torch.ones(self.k, *self.dim) * logvar_init, requires_grad=True)
        
        if logvar_bounds:
            self.logvar_bounding = True
            self.logvar_min, self.logvar_max = logvar_bounds
        else:
            self.logvar_bounding = False

    def forward(self) -> td.Distribution:
        mix_dist = td.Categorical(logits=self.mix_logits)
        if not self.logvar_bounding:
            logvars = self.logvars
        else:
            logvars = self._bounded_logvars()
        comp_dist = td.Independent(
            td.Normal(loc=self.means, scale=torch.exp(0.5 * logvars)),
            reinterpreted_batch_ndims=1
        )
        return td.MixtureSameFamily(mix_dist, comp_dist)
    
    def _bounded_logvars(self) -> torch.Tensor:
        return (
            self.logvar_min
                + (self.logvar_max - self.logvar_min)
                * torch.sigmoid(self.logvars)
        )

class VampBase(nn.Module):
    def __init__(
            self, dimensonality: Tuple[int], num_components: int,
            encoder: nn.Module|nn.Sequential, logit_init: float=3.0,
            pseudo_init: float=1.0, unflatten_flag=False,
            n_classes: int=0
        ) -> None:
        super().__init__()

        self.dim = dimensonality
        self.k = num_components
        self.pseudo_inputs = nn.Parameter(
            torch.rand(self.k, *self.dim) * pseudo_init,
            requires_grad=True
        )
        self.mix_logits = nn.Parameter(
            torch.ones(self.k) * logit_init, requires_grad=True)
        self.enc = encoder
        self.unflatten = unflatten_flag

        if n_classes > 0:
            labels = (
                torch.arange(n_classes)
                    .repeat((self.k + n_classes - 1)  // n_classes)
                    [:num_components]
            )
            self.register_buffer("pseudo_labels", labels.long())
        else:
            self.pseudo_labels = None
    
    def forward(self) -> td.Distribution:
        mix_dist = td.Categorical(logits=self.mix_logits)
        comp_dist = self.enc(self.pseudo_inputs, self.pseudo_labels)
        
        return td.MixtureSameFamily(mix_dist, comp_dist)

class DiscreteGraphBase(nn.Module):
    def __init__(self, dataset: InMemoryDataset, device: str="cpu") -> None:
        super().__init__()
        self.data = dataset
        self.n_graphs = len(dataset)
        self.device = device
        
        self.n_node_states = self.data.x.size(-1)
        self.n_edge_states = self.data.edge_attr.size(-1)
    
    def forward(
        self, return_dist: bool=False
    ) -> Tuple[torch.Tensor|torch.Tensor]|Tuple[td.Distribution, td.Distribution]:
        node_probs = self._make_node_base()
        edge_probs = self._make_edge_base()
        
        if return_dist:
            node_dist = td.Categorical(probs=node_probs)
            edge_dist = td.Categorical(probs=edge_probs)
            return node_dist, edge_dist

        return node_probs, edge_probs
    
    def sample(
        self, node_sample_shape: Tuple[int]=(1,),
        edge_sample_shape: Tuple[int]=(1,), y: torch.Tensor|None=None,
        to_one_hot: bool=True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_dist, edge_dist = self(return_dist=True)
        node_samples = node_dist.sample(node_sample_shape)
        edge_samples = edge_dist.sample(edge_sample_shape)

        if to_one_hot:
            node_samples = F.one_hot(
                node_samples, num_classes=self.n_node_states)
            edge_samples = F.one_hot(
                edge_samples, num_classes=self.n_edge_states)
        
        return node_samples, edge_samples
    
    def _make_node_base(self) -> torch.Tensor:
        raise NotImplementedError
    
    def _make_edge_base(self) -> torch.Tensor:
        raise NotImplementedError

class UniformGraphBase(DiscreteGraphBase):
    def _make_node_base(self) -> torch.Tensor:
        n_states = self.n_node_states
        return torch.tensor(
            [1/n_states]*n_states, dtype=torch.float32, device=self.device)
    
    def _make_edge_base(self) -> torch.Tensor:
        n_states = self.n_edge_states
        return torch.tensor(
            [1/n_states]*n_states, dtype=torch.float32, device=self.device)

class MarginalGraphBase(DiscreteGraphBase):
    def __init__(self, dataset: InMemoryDataset, device: str="cpu") -> None:
        super().__init__(dataset, device=device)
        self.register_buffer("node_probs", self._estimate_node_probs())
        self.register_buffer("edge_probs", self._estimate_edge_probs())

    def _make_node_base(self):
        return self.node_probs

    def _make_edge_base(self):
        return self.edge_probs

    def _estimate_node_probs(self):
        tokens = self.data.x.argmax(dim=-1)
        counts = tokens.bincount(minlength=self.n_node_states).float() + 1.0
        return (counts / counts.sum()).to(self.device)

    def _estimate_edge_probs(self):
        tokens = self.data.edge_attr.argmax(dim=-1)
        counts = tokens.bincount(minlength=self.n_edge_states).float() + 1.0
        return (counts / counts.sum()).to(self.device)
