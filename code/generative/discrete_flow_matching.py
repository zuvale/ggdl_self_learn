import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from typing import List, Tuple, Callable

from utilities.tensor_utils import broadcast_
from .base_dists import DiscreteGraphBase
from gnn_.utils import extract_triu_edge_data, mirror_triu_to_tril_again


class LinearDiscreteNoiser(nn.Module):
    def __init__(self, base_dist: DiscreteGraphBase) -> None:
        super().__init__()
        self.base = base_dist
    
    def forward(self, batch: Data, t: torch.Tensor) -> Data:
        batch = batch.clone()
        e_idx = batch.edge_index
        row, col = e_idx

        node_attr = batch.x
        t_node = t[batch.batch]

        triu_e_attr = extract_triu_edge_data(e_idx, batch.edge_attr)
        edge_graph = batch.batch[row[row < col]]
        t_edge = t[edge_graph]

        base_node_tokens, base_edge_tokens = self.base.sample(
            node_sample_shape=(t_node.shape[0],),
            edge_sample_shape=(t_edge.shape[0],)
        )
        
        keep_nodes = broadcast_((torch.rand_like(t_node) < t_node), node_attr)
        keep_edges = broadcast_((torch.rand_like(t_edge) < t_edge), triu_e_attr)

        new_node_attr = torch.where(keep_nodes, node_attr, base_node_tokens)
        new_triu_e_attr = torch.where(keep_edges, triu_e_attr, base_edge_tokens)

        batch.x = new_node_attr
        batch.edge_attr = mirror_triu_to_tril_again(
            e_idx, batch.edge_attr, new_triu_e_attr, batch.num_nodes)
        
        return batch

class LinearDiscreteRateMatrix(nn.Module):
    EPS: float=1e-8
    def __init__(
        self, base_dist: nn.Module, n_node_types: int, n_edge_types: int
    ) -> None:
        super().__init__()
        self.base = base_dist
        self.n_node_types = n_node_types
        self.n_edge_types = n_edge_types
    
    def forward(
        self, current_batch: Data,
        graph_final_token_pred: Tuple[torch.Tensor, torch.Tensor],
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 0. Define Helper Variables
        batch = current_batch.clone()
        e_idx = batch.edge_index
        row, col = e_idx
        upper_graph = batch.batch[row[row < col]]

        node_current_tokens, edge_current_tokens = batch.x, batch.edge_attr
        node_final_tokens, edge_final_tokens = graph_final_token_pred

        # 1. Broadcast Time and Extract Upper-Triangular Edge Data
        t_node = t[batch.batch][:, None]
        triu_e_current = extract_triu_edge_data(
            e_idx, edge_current_tokens)
        triu_e_final = extract_triu_edge_data(
            e_idx, edge_final_tokens)
        t_edge = t[upper_graph][:, None]

        # 2.
        d_pt_n, d_pt_e = self.prob_path_rate(
            node_final_tokens, triu_e_final)
        d_pt_n_at_current = d_pt_n.gather(
            -1, node_current_tokens.argmax(dim=-1)[:, None])
        d_pt_e_at_current = d_pt_e.gather(
            -1, triu_e_current.argmax(dim=-1)[:, None])

        # 3.
        pt_n, pt_e = self.prob_path_mass(
            node_final_tokens, triu_e_final, t_node, t_edge)
        n_pos_token_prob = pt_n.count_nonzero(dim=-1)[:, None]
        e_pos_token_prob = pt_e.count_nonzero(dim=-1)[:, None]
        pt_n_at_current = pt_n.gather(
            -1, node_current_tokens.argmax(dim=-1)[:, None])
        pt_e_at_current = pt_e.gather(
            -1, triu_e_current.argmax(dim=-1)[:, None])

        # 4. Calculate KFE-Based Transitional Rates
        rate_mat_nodes = (
            F.relu(d_pt_n - d_pt_n_at_current)
                / (n_pos_token_prob * pt_n_at_current).clamp_min(self.EPS)
        )
        triu_r_e = (
            F.relu(d_pt_e - d_pt_e_at_current)
                / (e_pos_token_prob * pt_e_at_current).clamp_min(self.EPS)
        )

        rate_mat_edges = mirror_triu_to_tril_again(
            e_idx, edge_current_tokens.float(), triu_r_e, batch.num_nodes)
        return rate_mat_nodes, rate_mat_edges

    def prob_path_mass(
        self, n_hat_final: torch.Tensor, e_hat_final: torch.Tensor,
        t_node: torch.Tensor, t_edge: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_init, e_init = self.base(return_dist=False)

        pt_n = (
            t_node*F.one_hot(n_hat_final, num_classes=self.n_node_types)
                + (1 - t_node)*n_init.unsqueeze(0)
        )
        pt_e = (
            t_edge*F.one_hot(e_hat_final, num_classes=self.n_edge_types)
                + (1 - t_edge)*e_init.unsqueeze(0)
        )

        return pt_n, pt_e
    
    def prob_path_rate(
        self, n_hat_final: torch.Tensor, e_hat_final: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_init, e_init = self.base(return_dist=False)

        d_pt_n = (
            F.one_hot(n_hat_final, num_classes=self.n_node_types)
                - n_init.unsqueeze(0)
        )
        d_pt_e = (
            F.one_hot(e_hat_final, num_classes=self.n_edge_types)
                - e_init.unsqueeze(0)
        )

        return d_pt_n, d_pt_e

class CTMCSampler(nn.Module):
    EPS: float=1e-8
    def __init__(
        self, base_dist: nn.Module, rate_matrix : nn.Module,
        node_classes: int, edge_classes: int, max_nodes: int,
        update_method: str="euler", device: str="cpu",
        bond_types: List["str"]=["single", "double", "triple", "no_bond"]
    ) -> None:
        super().__init__()
        self.base = base_dist
        self.rate_matrix = rate_matrix

        self.nc_nodes, self.nc_edges = node_classes, edge_classes
        self.max_n_nodes = max_nodes

        if update_method == "euler":
            self.update_step = lambda d, b, t, h, u, **kwargs: self.euler_step(
                d, b, t, h, u, self.nc_nodes, self.nc_edges, **kwargs)
        
        self.device = device

        self.bond_types = bond_types
    
    @torch.inference_mode
    def forward(
        self, denoiser: nn.Module,
        sample_shape: Tuple[int]=(1,), n_steps: int=100,
        y: torch.Tensor|None=None, show_path: bool=False,
        t_eval: Tuple[int|float, int|float]=(0., 1.), **kwargs
    ) -> Data|List[Data]:
        batch = self.initial_samples(
            self.base, sample_shape[0], self.max_n_nodes, y=y,
            device=self.device
        )
        if show_path:
            path_samples = [batch]
        
        t0, t_end = t_eval
        t = torch.full((batch.num_graphs,), t0, device=self.device)
        h = torch.full(
            (batch.num_graphs,), (t_end - t0) / n_steps, device=self.device)
        for k in range(n_steps):
            batch, t = self.update_step(
                denoiser, batch, t, h, self.compute_step_probs, **kwargs)
            if show_path:
                path_samples.append(batch)
        
        if show_path:
            return path_samples
        
        return batch

    def compute_step_probs(
        self, denoiser: nn.Module, batch: Data, t: torch.Tensor,
        h: torch.Tensor, bond_order_bias: Tuple[float, float, float]|None=None,
        exit_cap: float|None=None, temp_scales: Tuple[float, float]=(1., 1.)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = batch.clone()
        e_idx, n_nodes = batch.edge_index, batch.num_nodes
        row, col = e_idx
        upper_graph = batch.batch[row[row < col]]
        h_node, h_edge = h[batch.batch][:, None], h[upper_graph][:, None]

        node_logits, edge_logits = denoiser(batch, t)
        if bond_order_bias:
            dbl_b, tpl_b, ne_b = bond_order_bias
            edge_logits[:, self.bond_types.index("double")] += dbl_b
            edge_logits[:, self.bond_types.index("triple")] += tpl_b
            edge_logits[:, self.bond_types.index("no_bond")] += ne_b
        triu_e_logits = extract_triu_edge_data(e_idx, edge_logits)
        node_final, triu_e_final = self.endpoint_sampling(
            node_logits, triu_e_logits, node_temp=temp_scales[0],
            edge_temp=temp_scales[0]
        )
        edge_final = mirror_triu_to_tril_again(
            e_idx, edge_logits.argmax(dim=1), triu_e_final, n_nodes)
        
        node_rate, edge_rate = self.rate_matrix(
            batch, (node_final, edge_final), t)
        
        step_prob_node = h_node * node_rate
        step_prob_node = self.normalize_step_probs(
            batch.x, step_prob_node, exit_cap=exit_cap)
        triu_e_prob = h_edge * extract_triu_edge_data(e_idx, edge_rate)
        triu_e_token = extract_triu_edge_data(e_idx, batch.edge_attr)
        triu_e_prob = self.normalize_step_probs(
            triu_e_token, triu_e_prob, exit_cap=exit_cap)
        step_prob_edge = mirror_triu_to_tril_again(
            e_idx, edge_logits, triu_e_prob, n_nodes)
        
        return step_prob_node, step_prob_edge
    
    def normalize_step_probs(
        self, token: torch.Tensor, step_probs: torch.Tensor,
        exit_cap: float|None=None
    ) -> torch.Tensor:
        if token.dim() == 2:
            token = token.argmax(dim=-1)

        offdiag_mass = step_probs.sum(dim=-1, keepdim=True)
        if exit_cap:
            scale = torch.clamp(
                exit_cap / offdiag_mass.clamp_min(self.EPS),
                max=1.0,
            )
            step_probs = step_probs * scale
            offdiag_mass = step_probs.sum(dim=-1, keepdim=True)

        stay_prob = (1.0 - offdiag_mass).clamp_min(0.0)
        step_probs = step_probs.scatter(
            dim=-1, index=token[:, None], src=stay_prob)

        step_probs = step_probs.clamp_min(self.EPS)
        step_probs = step_probs / step_probs.sum(dim=-1, keepdim=True)

        return step_probs

    @staticmethod
    def initial_samples(
        base_dist: nn.Module, batch_size: int, max_n_nodes: int,
        y: torch.Tensor|None=None, device: str="cpu"
    ) -> Data:
        data_list = []

        row, col = torch.triu_indices(
            max_n_nodes, max_n_nodes, offset=1, device=device)
        triu_edge_index = torch.stack([row, col], dim=0)
        num_edges = triu_edge_index.size(1)
        edge_index = torch.cat(
            [triu_edge_index, triu_edge_index.flip(0)], dim=1)

        for i in range(batch_size):
            x, upper_edge_attr = base_dist.sample(
                node_sample_shape=(max_n_nodes,), edge_sample_shape=(num_edges,)
            )
            
            edge_attr = torch.cat([upper_edge_attr, upper_edge_attr], dim=0)
            data = Data(
                x=x,
                edge_index=edge_index.clone(),
                edge_attr=edge_attr,
                y=y[i].view(1) if y is not None else None,
                num_nodes=max_n_nodes,
            )

            data_list.append(data)

        return Batch.from_data_list(data_list).to(device)
    
    @staticmethod
    def endpoint_sampling(
        node_logits: torch.Tensor, edge_logits: torch.Tensor,
        node_temp: float=1.0, edge_temp: float=1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_final = (
            torch.distributions.Categorical(logits=node_logits / node_temp)
                .sample()
        )
        edge_final = (
            torch.distributions.Categorical(logits=edge_logits / edge_temp)
                .sample()
        )
        
        return node_final, edge_final

    @staticmethod
    def euler_step(
        denoiser: nn.Module, batch: Data, t: torch.Tensor, h: torch.Tensor,
        fun: Callable, nc_node: int, nc_edge: int, **kwargs
    ) -> Tuple[torch.Tensor|torch.Tensor]:
        batch = batch.clone()
        e_idx, n_nodes = batch.edge_index, batch.num_nodes

        P_X, P_E = fun(denoiser, batch, t, h, **kwargs)
        t = t + h

        triu_P_E = extract_triu_edge_data(e_idx, P_E)
        node_tokens = torch.distributions.Categorical(probs=P_X).sample()
        triu_e_tokens = torch.distributions.Categorical(probs=triu_P_E).sample()
        edge_tokens = mirror_triu_to_tril_again(
            e_idx, P_E.argmax(dim=1), triu_e_tokens, n_nodes)

        batch = batch.clone()
        batch.x = F.one_hot(node_tokens, num_classes=nc_node)
        batch.edge_attr = F.one_hot(edge_tokens, num_classes=nc_edge)

        return batch, t

class DeFoG(nn.Module):
    def __init__(
        self, base_dist: nn.Module, noising_process: nn.Module,
          denoiser: nn.Module, sampler: nn.Module, node_tokens: int,
          edge_tokens: int, max_n_nodes: int=35
    ) -> None:
        super().__init__()

        self.base = base_dist
        self.noiser = noising_process
        self.denoiser = denoiser
        self.sampler = sampler
        self.node_tokens = node_tokens
        self.edge_tolens = edge_tokens
        self.max_n_nodes = max_n_nodes
    
    def forward(
        self, batch: Data, edge_loss_weight: float=1.0,
        edge_weights: torch.Tensor=torch.ones(4)
    ) -> torch.Tensor:
        t = torch.rand(len(batch), device=batch.x.device)
        noisy_batch = self.noiser(batch, t)

        node_logits, edge_logits = self.denoiser(noisy_batch, t)

        node_loss = F.cross_entropy(node_logits, batch.x.float())
        edge_weights = edge_weights.to(batch.x.device)
        triu_e_attr = extract_triu_edge_data(batch.edge_index, batch.edge_attr)
        triu_e_logits = extract_triu_edge_data(batch.edge_index, edge_logits)
        edge_loss = F.cross_entropy(
            triu_e_logits, triu_e_attr.float(), weight=edge_weights)

        loss = node_loss + edge_loss_weight * edge_loss

        return loss
    
    @torch.inference_mode()
    def sample(
        self, sample_shape=(1,), n_steps: int=100, y: torch.Tensor|None=None,
        show_path: bool=False,
        bond_order_bias: Tuple[float, float, float]=(0., 0., 0.),
        exit_cap: float=0.5, temp_scales: Tuple[float, float]=(1., 1.)
    ) -> torch.Tensor|Tuple[torch.Tensor, torch.Tensor]:
        return self.sampler(
            self.denoiser, sample_shape, y=y, n_steps=n_steps,
            show_path=show_path, bond_order_bias=bond_order_bias,
            exit_cap=exit_cap, temp_scales=temp_scales
        )