import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from torch_geometric.data import Data, Batch
from typing import Tuple, Callable, List

from utilities.tensor_utils import broadcast_
from .base_dists import graph_initial_samples
from gnn_.utils import extract_triu_edge_data, mirror_triu_to_tril_again


class ProbabilityPath(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(
        self, t: torch.Tensor, x1: torch.Tensor, base_dist: td.Distribution,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    
    def _broadcast(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Broadcast some parameter to data shape.
        """
        return v.view(v.size(0), *([1]*(x.dim() - 1)))
    
    def _extract(
        self, v: torch.Tensor, t: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract correct dimensions from time-dependent parameter and broadcast
        to data shape.
        """
        v = v.gather(0, t)
        return self._broadcast(v, x)

    @staticmethod
    def _sample_base(
        base_dist: td.Distribution, x1: torch.Tensor) -> torch.Tensor:
        return base_dist.sample(
            (x1.shape[0],)).to(device=x1.device, dtype=x1.dtype)

class GaussianProbabilityPath(ProbabilityPath):
    def __init__(self, schedule: nn.Module):
        super().__init__()
        self.schedule = schedule

    def forward(
        self, t: torch.Tensor, x1: torch.Tensor, base_dist: td.Distribution,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t = self._broadcast(t, x1)
        epsilon = self._sample_base(base_dist, x1)

        alpha, beta = self.schedule.alpha(t), self.schedule.beta(t)
        alpha_dt, beta_dt = self.schedule.alpha_dt(t), self.schedule.beta_dt(t)

        x_t = alpha*x1 + beta*epsilon
        u_t = alpha_dt*x1 + beta_dt*epsilon
        return x_t, u_t

class LinearGaussianSchedule(nn.Module):
    """
    Linear OT interpolation viewed as a special case of a Gaussian path
    """
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return t
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - t
    def alpha_dt(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)
    def beta_dt(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)

class SqrtGaussianSchedule(nn.Module):
    def __init__(self, eps: float=1e-4) -> None:
        super().__init__()
        self.eps = eps

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return t
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp(1 - t, min=self.eps))
    def alpha_dt(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)
    def beta_dt(self, t: torch.Tensor) -> torch.Tensor:
        return -0.5 / torch.sqrt(torch.clamp(1 - t, min=self.eps))

class CatFlowInterpolant(nn.Module):
    def __init__(self, base_dist: nn.Module, max_n_nodes: int) -> None:
        super().__init__()
        self.base = base_dist
        self.max_n_nodes = max_n_nodes
    
    def forward(self, batch: Data, t: torch.Tensor) -> Data:
        base_batch = graph_initial_samples(
            self.base, batch.num_graphs, self.max_n_nodes, batch.y,
            device=batch.x.device
        )

        return self.interpolate(base_batch, batch, t)

    @staticmethod
    def interpolate(src_batch, tgt_batch, t: torch.Tensor) -> Data:
        intpol_batch = tgt_batch.clone()

        e_idx = intpol_batch.edge_index
        row, col = e_idx
        upper_graph = intpol_batch.batch[row[row < col]]
        t_node, t_edge = t[intpol_batch.batch][:, None], t[upper_graph][:, None]

        intpol_node = t_node * tgt_batch.x + (1 - t_node) * src_batch.x

        triu_e_src = extract_triu_edge_data(e_idx, src_batch.edge_attr)
        triu_e_tgt = extract_triu_edge_data(e_idx, tgt_batch.edge_attr)
        triu_intpol_e = t_edge * triu_e_tgt + (1 - t_edge) * triu_e_src
        intpol_edge = mirror_triu_to_tril_again(
            e_idx, intpol_batch.edge_attr.float(), triu_intpol_e,
            intpol_batch.num_nodes
        )
        
        intpol_batch.x = intpol_node
        intpol_batch.edge_attr = intpol_edge
        return intpol_batch

class ODESampler(nn.Module):
    def __init__(
        self, base_dist: nn.Module, update_method: str="euler",
        device: str="cpu"
    ) -> None:
        super().__init__()
        self.base = base_dist

        if update_method == "euler":
            self.update_step = self.euler_step
        
        self.device = device

    @torch.inference_mode
    def forward(
        self, velocity_field: nn.Module,
        sample_shape: Tuple[int]=(1,), n_steps: int=100,
        y: torch.Tensor|None=None, show_path: bool=False,
        t_eval: Tuple[int|float, int|float]=(0., 1.), **kwargs
    ) -> torch.Tensor|List[torch.Tensor]:
        x = self.base().sample(sample_shape)
        if show_path:
            path_samples = [x]

        t0, t_end = t_eval
        t = torch.full(sample_shape, t0, device=self.device)
        h = torch.full(
            sample_shape, (t_end - t0) / n_steps, device=self.device)
        t, h = broadcast_(t, x), broadcast_(h, x)
        for k in range(n_steps):
            x, t = self.update_step(velocity_field, x, t, h, y=y, **kwargs)
            if show_path:
                path_samples.append(x)

        if show_path:
            return torch.stack(path_samples, dim=0)
        return x
    
    @staticmethod
    def euler_step(
        velocity_field: nn.Module, x: torch.Tensor, t: torch.Tensor,
        h: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + h * velocity_field(x, t, **kwargs)
        t = t + h
        return x, t

class ShortcutODESampler(ODESampler):
    @staticmethod
    def euler_step(
        velocity_field: nn.Module, x: torch.Tensor, t: torch.Tensor,
        h: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + h * velocity_field(x, t, h, **kwargs)
        t = t + h
        return x, t

class CatFlowODESampler(nn.Module):
    EPS: float=1e-8
    def __init__(
        self, base_dist: nn.Module, node_classes: int, edge_classes: int,
        max_nodes: int, update_method: str="euler", device: str="cpu",
        bond_types: List["str"]=["single", "double", "triple", "no_bond"]
    ) -> None:
        super().__init__()
        self.base = base_dist

        self.nc_nodes, self.nc_edges = node_classes, edge_classes
        self.max_n_nodes = max_nodes

        if update_method == "euler":
            self.update_step = self.euler_step

        self.device = device
        self.bond_types = bond_types
    
    @torch.inference_mode
    def forward(
        self, denoiser: nn.Module,
        sample_shape: Tuple[int]=(1,), n_steps: int=100,
        y: torch.Tensor|None=None, show_path: bool=False,
        t_eval: Tuple[int|float, int|float]=(0., 1.), **kwargs
    ) -> Data|List[Data]:
        batch = graph_initial_samples(
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
            batch = self.normalize_simplex(batch, eps=self.EPS)
            if show_path:
                path_samples.append(self.discretize_graph_probs(
                    batch, self.nc_nodes, self.nc_edges))
        if show_path:
            return path_samples
        
        return self.discretize_graph_probs(batch, self.nc_nodes, self.nc_edges)
    
    def compute_step_probs(
        self, denoiser: nn.Module, batch: Data, t: torch.Tensor,
        *args: torch.Tensor,
        bond_order_bias: Tuple[float, float, float]|None=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_logits, edge_logits = denoiser(batch, t, *args)
        if bond_order_bias:
            dbl_b, tpl_b, ne_b = bond_order_bias
            edge_logits[:, self.bond_types.index("double")] += dbl_b
            edge_logits[:, self.bond_types.index("triple")] += tpl_b
            edge_logits[:, self.bond_types.index("no_bond")] += ne_b
        
        return node_logits, edge_logits
    
    @staticmethod
    def discretize_graph_probs(
        batch: Data, nc_node: int, nc_edge: int) -> Data:
        batch = batch.clone()

        batch.x = batch.x.argmax(dim=1)
        batch.edge_attr = batch.edge_attr.argmax(dim=1)
        batch.x = F.one_hot(batch.x, num_classes=nc_node)
        batch.edge_attr = F.one_hot(batch.edge_attr, num_classes=nc_edge)

        return batch
    
    @staticmethod
    def euler_step(
        denoiser: nn.Module, batch: Data, t: torch.Tensor, h: torch.Tensor,
        fun: Callable, eps=1e-3, **kwargs
    ) -> Tuple[Data|torch.Tensor]:
        batch = batch.clone()
        e_idx, n_nodes = batch.edge_index, batch.num_nodes
        row, col = e_idx
        upper_graph = batch.batch[row[row < col]]
        t_node, t_edge = t[batch.batch][:, None], t[upper_graph][:, None]
        h_node, h_edge = h[batch.batch][:, None], h[upper_graph][:, None]

        P_X, P_E = fun(denoiser, batch, t, **kwargs)
        t = t + h

        mu_X = F.softmax(P_X, dim=-1)
        v_P = (mu_X - batch.x) / (1 - t_node).clamp_min(eps)

        triu_P_E = extract_triu_edge_data(e_idx, P_E)
        triu_e_attr = extract_triu_edge_data(e_idx, batch.edge_attr)
        triu_mu_E = F.softmax(triu_P_E, dim=-1)
        triu_v_E = (triu_mu_E - triu_e_attr) / (1 - t_edge).clamp_min(eps)
        

        batch.x = batch.x + h_node * v_P
        triu_e_next = triu_e_attr + h_edge * triu_v_E
        batch.edge_attr = mirror_triu_to_tril_again(
            e_idx, batch.edge_attr.float(), triu_e_next, n_nodes)
        return batch, t
    
    @staticmethod
    def normalize_simplex(batch, eps: float=1e-8) -> Data:
        batch = batch.clone()
        e_idx, n_nodes = batch.edge_index, batch.num_nodes

        x = batch.x.clamp_min(eps)
        x = x / x.sum(dim=-1, keepdim=True)

        triu_e_attr = extract_triu_edge_data(e_idx, batch.edge_attr)
        triu_e_attr = triu_e_attr.clamp_min(eps)
        triu_e_attr = triu_e_attr / triu_e_attr.sum(dim=-1, keepdim=True)
        edge_attr = mirror_triu_to_tril_again(
            e_idx, batch.edge_attr, triu_e_attr, n_nodes)
        
        batch.x, batch.edge_attr = x, edge_attr
        return batch

class ShortcutCatFlowODESampler(CatFlowODESampler):
    @staticmethod
    def euler_step(
        denoiser: nn.Module, batch: Data, t: torch.Tensor, h: torch.Tensor,
        fun: Callable, eps=1e-3, **kwargs
    ) -> Tuple[Data|torch.Tensor]:
        batch = batch.clone()
        e_idx, n_nodes = batch.edge_index, batch.num_nodes
        row, col = e_idx
        upper_graph = batch.batch[row[row < col]]
        t_node, t_edge = t[batch.batch][:, None], t[upper_graph][:, None]
        h_node, h_edge = h[batch.batch][:, None], h[upper_graph][:, None]

        P_X, P_E = fun(denoiser, batch, t, h, **kwargs)
        t = t + h

        mu_X = F.softmax(P_X, dim=-1)
        v_P = (mu_X - batch.x) / (1 - t_node).clamp_min(eps)

        triu_P_E = extract_triu_edge_data(e_idx, P_E)
        triu_e_attr = extract_triu_edge_data(e_idx, batch.edge_attr)
        triu_mu_E = F.softmax(triu_P_E, dim=-1)
        triu_v_E = (triu_mu_E - triu_e_attr) / (1 - t_edge).clamp_min(eps)
        

        batch.x = batch.x + h_node * v_P
        triu_e_next = triu_e_attr + h_edge * triu_v_E
        batch.edge_attr = mirror_triu_to_tril_again(
            e_idx, batch.edge_attr.float(), triu_e_next, n_nodes)
        return batch, t

class FlowMatchingModel(nn.Module):
    def __init__(
        self, velocity_field: nn.Module, prob_path: nn.Module,
        base_dist: nn.Module, sampler: nn.Module, device: str="cpu",
        time_scale: Tuple[int|float, int|float]=(0., 1.)
    ) -> None:
        super().__init__()

        self.device = device
        self.base = base_dist.to(self.device)
        self.vf = velocity_field
        self.path = prob_path.to(self.device)
        self.sampler = sampler
        self.time_scale = time_scale
    
    def forward(
        self, x1: torch.Tensor, y: torch.Tensor|None=None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.rand((x1.shape[0], 1), device=self.device)

        cond_path_sample, cond_velocity_sample = self.path(t, x1, self.base())
        velocity_prediction = self.vf(
            cond_path_sample.to(self.device), t, y=y, **kwargs)

        return velocity_prediction, cond_velocity_sample

    @torch.inference_mode()
    def sample(
        self, sample_shape=(1,), n_steps: int=100, y: torch.Tensor|None=None,
        show_path: bool=False, **kwargs
    ) -> torch.Tensor:
        return self.sampler(
            self.vf, sample_shape, y=y, n_steps=n_steps,
            show_path=show_path, **kwargs
        )

class ShortCutFM(FlowMatchingModel):
    SHORTCUT_SIZES: List[float]=torch.tensor(
        [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2], dtype=torch.float32)

    def forward(
        self, x1: torch.Tensor, y: torch.Tensor|None=None,
        consist_ratio: float=0.25, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # get the indices for splitting the batch
        batch_size = x1.size(0)
        fm_idx, consist_idx = self.split_batch_idxes(
            batch_size, consist_ratio, device=x1.device)
        x1_fm = x1[fm_idx]
        x1_consist = x1[consist_idx]
        if y is not None:
            y_fm, y_consist = y[fm_idx], y[consist_idx]
        else:
            y_fm, y_consist = None, None

        # sample time and step-size separetely for FM and consistency loss
        t_fm = torch.rand((len(fm_idx), 1), device=self.device)
        h_fm = torch.zeros_like(t_fm)
        t_consist, h_consist = self.sample_shortcut_t_h(
            self.SHORTCUT_SIZES.to(x1.device), len(consist_idx))
        
        # get the conditional path samples
        fm_path, fm_velocity = self.path(t_fm, x1_fm, self.base())
        consist_path, consist_velocity = self.path(
            t_consist, x1_consist, self.base())

        # the standard flow-matching prediction
        fm_pred = self.vf(
            fm_path.to(self.device), t_fm, h_fm, y=y_fm,
            **kwargs
        )
        
        # self-consistency path
        # first, small prediction
        bootstrap_1 = self.vf(
            consist_path.to(self.device), t_consist, h_consist, y=y_consist,
            **kwargs
        )
        # follow VF-ODE (Euler step)
        bootstrap_step = (
            consist_path + broadcast_(h_consist, bootstrap_1) * bootstrap_1)
        # next small prediction
        bootstrap_2 = self.vf(
            bootstrap_step, t_consist + h_consist, h_consist, y=y_consist,
            **kwargs
        )
        # detach self-consistency target
        consist_velocity = (bootstrap_1 + bootstrap_2).detach() / 2
        # make large prediction with double step-size
        consist_pred = self.vf(
            consist_path.to(self.device), t_consist, 2*h_consist, y=y_consist,
            **kwargs
        )
        
        # concatenate again
        pred = torch.cat((fm_pred, consist_pred), dim=0)
        target = torch.cat((fm_velocity, consist_velocity), dim=0)
        return pred, target
    
    @staticmethod
    def sample_shortcut_t_h(
        sc_sizes: torch.Tensor, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_idx = torch.randint(
            low=0, high=len(sc_sizes), size=(batch_size,),
            device=sc_sizes.device
        )
        h = sc_sizes[h_idx]

        n_intervals = torch.round((1.0 - 2.0 * h) / h).long() + 1
        k = (
            torch.rand(batch_size, device=sc_sizes.device) * n_intervals
        ).long()
        t = k.float() * h

        return t, h

    @staticmethod
    def split_batch_idxes(
        batch_size: int, consist_ratio: int, device: str="cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_consist = int(round(consist_ratio * batch_size))

        perm = torch.randperm(batch_size, device=device)
        fm_idx = perm[n_consist:]
        consist_idx = perm[:n_consist]

        return fm_idx, consist_idx

class CatFlow_old(nn.Module):
    def __init__(
        self, base_dist: nn.Module, interpolant: nn.Module,
          denoiser: nn.Module, sampler: nn.Module, node_tokens: int,
          edge_tokens: int, max_n_nodes: int=35
    ) -> None:
        super().__init__()

        self.base = base_dist
        self.interpolant = interpolant
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
        intpol_batch = self.interpolant(batch, t)

        node_logits, edge_logits = self.denoiser(intpol_batch, t)

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
    ) -> torch.Tensor|Tuple[torch.Tensor, torch.Tensor]:
        return self.sampler(
            self.denoiser, sample_shape, y=y, n_steps=n_steps,
            show_path=show_path, bond_order_bias=bond_order_bias
        )

class CatFlow(FlowMatchingModel):
    def __init__(
        self, base_dist: nn.Module, interpolant: nn.Module,
        denoiser: nn.Module, sampler: nn.Module, node_tokens: int,
        edge_tokens: int, max_n_nodes: int=35, device: str="cpu",
        time_scale: Tuple[int|float, int|float]=(0., 1.)
    ) -> None:
        super().__init__(
            denoiser, interpolant, base_dist, sampler, device=device,
            time_scale=time_scale
        )

        self.node_tokens = node_tokens
        self.edge_tolens = edge_tokens
        self.max_n_nodes = max_n_nodes

    def forward(
        self, batch: Data, edge_loss_weight: float=1.0,
        edge_weights: torch.Tensor=torch.ones(4)
    ) -> torch.Tensor:
        t = torch.rand(len(batch), device=batch.x.device)
        intpol_batch = self.path(batch, t)

        node_logits, edge_logits = self.vf(intpol_batch, t)

        node_loss = F.cross_entropy(node_logits, batch.x.float())
        edge_weights = edge_weights.to(batch.x.device)
        triu_e_attr = extract_triu_edge_data(batch.edge_index, batch.edge_attr)
        triu_e_logits = extract_triu_edge_data(batch.edge_index, edge_logits)
        edge_loss = F.cross_entropy(
            triu_e_logits, triu_e_attr.float(), weight=edge_weights)

        loss = node_loss + edge_loss_weight * edge_loss

        return loss

class ShortcutCatFlow(CatFlow, ShortCutFM):
    def forward(
        self, batch: Data, consist_ratio: float=0.25,
        edge_loss_weight: float=1.0, edge_weights: torch.Tensor=torch.ones(4),
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # get the indices for splitting the batch
        fm_idx, consist_idx = self.split_batch_idxes(
            batch.num_graphs, consist_ratio, device=batch.x.device)
        # split the batch into the two paths
        fm_batch = self.subset_batch(batch, fm_idx)
        consist_batch = self.subset_batch(batch, consist_idx)

        # get graph-based indices for the two paths
        e_idx_fm, e_idx_consist = fm_batch.edge_index, consist_batch.edge_index
        (row_fm, col_fm), (row_consist, col_consist) = e_idx_fm, e_idx_consist
        upper_graph_fm = fm_batch.batch[row_fm[row_fm < col_fm]]
        upper_graph_consist = (
            consist_batch.batch[row_consist[row_consist < col_consist]])

        # sample time and step-size separetely for FM and consistency loss
        edge_weights = edge_weights.to(batch.x.device)
        t_fm = torch.rand(len(fm_idx), device=self.device)
        h_fm = torch.zeros_like(t_fm)
        t_consist, h_consist = self.sample_shortcut_t_h(
            self.SHORTCUT_SIZES.to(batch.x.device), len(consist_idx)
        )

        t_node_consist = t_consist[consist_batch.batch][:, None]
        t_edge_consist = t_consist[upper_graph_consist][:, None]
        h_node_consist = h_consist[consist_batch.batch][:, None]
        h_edge_consist = h_consist[upper_graph_consist][:, None]

        intpol_batch_fm = self.path(fm_batch, t_fm)
        intpol_batch_consist = self.path(consist_batch, t_consist)

        node_logits_fm, edge_logits_fm = self.vf(
            intpol_batch_fm, t_fm, h_fm, **kwargs)
        triu_e_logits_fm = extract_triu_edge_data(e_idx_fm, edge_logits_fm)

        node_logits_boot1, edge_logits_boot1 = self.vf(
            intpol_batch_consist, t_consist, h_consist, **kwargs)
        triu_e_logits_boot1 = extract_triu_edge_data(
            e_idx_consist, edge_logits_boot1)

        triu_e_attr_consist = extract_triu_edge_data(
            e_idx_consist, intpol_batch_consist.edge_attr)

        v_node_boot1 = self.get_velocity(
            node_logits_boot1, intpol_batch_consist.x, t_node_consist)
        v_triu_e_boot1 = self.get_velocity(
            triu_e_logits_boot1, triu_e_attr_consist, t_edge_consist)

        boot2_input = intpol_batch_consist.clone()
        boot2_input.x = intpol_batch_consist.x + h_node_consist * v_node_boot1

        triu_e_bootstep = triu_e_attr_consist + h_edge_consist * v_triu_e_boot1
        boot2_input.edge_attr = mirror_triu_to_tril_again(
            e_idx_consist,
            intpol_batch_consist.edge_attr,
            triu_e_bootstep,
            intpol_batch_consist.num_nodes,
        )
        boot2_input = self.sampler.normalize_simplex(boot2_input)

        t_next = t_consist + h_consist
        t_node_next = t_next[boot2_input.batch][:, None]
        t_edge_next = t_next[upper_graph_consist][:, None]

        node_logits_boot2, edge_logits_boot2 = self.vf(
            boot2_input, t_next, h_consist, **kwargs)
        triu_e_logits_boot2 = extract_triu_edge_data(
            e_idx_consist, edge_logits_boot2)
        triu_e_bootstep = extract_triu_edge_data(
            e_idx_consist, boot2_input.edge_attr)

        v_node_boot2 = self.get_velocity(
            node_logits_boot2, boot2_input.x, t_node_next)
        v_triu_e_boot2 = self.get_velocity(
            triu_e_logits_boot2, triu_e_bootstep, t_edge_next)

        tgt_v_node = (v_node_boot1 + v_node_boot2).detach() / 2
        tgt_v_triu_e = (v_triu_e_boot1 + v_triu_e_boot2).detach() / 2

        node_logits_consist, edge_logits_consist = self.vf(
            intpol_batch_consist, t_consist, 2 * h_consist, **kwargs)
        triu_e_logits_consist = extract_triu_edge_data(
            e_idx_consist, edge_logits_consist)

        pred_v_node = self.get_velocity(
            node_logits_consist, intpol_batch_consist.x, t_node_consist)
        pred_v_triu_e = self.get_velocity(
            triu_e_logits_consist, triu_e_attr_consist, t_edge_consist)

        triu_e_fm_tgt = extract_triu_edge_data(
            e_idx_fm, fm_batch.edge_attr)

        fm_loss = (
            F.cross_entropy(node_logits_fm, fm_batch.x.float())
            + edge_loss_weight * F.cross_entropy(
                triu_e_logits_fm, triu_e_fm_tgt.float(), weight=edge_weights)
        )

        consist_loss = (
            F.mse_loss(pred_v_node, tgt_v_node)
            + edge_loss_weight * F.mse_loss(pred_v_triu_e, tgt_v_triu_e)
        )

        return fm_loss + consist_loss

    @staticmethod
    def subset_batch(batch: Data, idx: torch.Tensor) -> Data:
        idx = idx.detach().cpu().tolist()
        data_list = [batch.get_example(i) for i in idx]
        return Batch.from_data_list(data_list).to(batch.x.device)
    
    @staticmethod
    def get_velocity(
        logits: torch.Tensor, feats: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return (F.softmax(logits, dim=-1) - feats) / (1 - t).clamp_min(1e-3)