import torch
import torch.nn as nn
import torch.distributions as td
from typing import Tuple, Callable


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

class FlowMatchingModel(nn.Module):
    def __init__(
        self, velocity_field: nn.Module, prob_path: nn.Module,
        base_dist: nn.Module, solver: Callable, device: str="cpu"
    ) -> None:
        super().__init__()

        self.device = device
        self.p = base_dist.to(self.device)
        self.vf = velocity_field
        self.path = prob_path.to(self.device)
        self.solver = solver
    
    def forward(
        self, x1: torch.Tensor, y: torch.Tensor|None=None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.rand((x1.shape[0], 1), device=self.device)

        cond_path_sample, cond_velocity_sample = self.path(t, x1, self.p())
        velocity_prediction = self.vf(
            cond_path_sample.to(self.device), t, y=y, **kwargs)

        return velocity_prediction, cond_velocity_sample

    @torch.inference_mode()
    def sample(
        self, sample_shape=(1,), n_steps: int=100, y: torch.Tensor|None=None,
        show_path: bool=False, ode_solver: None|Callable=None, **kwargs
    ) -> torch.Tensor|Tuple[torch.Tensor, torch.Tensor]:
        x_init = self.p().sample(sample_shape)
        solver = self.solver if not ode_solver else ode_solver

        def vf_ode(x, t):
            return self.vf(x, t, y=y, **kwargs)

        x_path, ts = solver(
            vf_ode, [0., 1.], x_init, n_steps, device=self.device)
        if not show_path:
            return x_path[-1]
        else:
            return x_path, ts