import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class NoiseSchedule(nn.Module):
    def __init__(self, n_timesteps: int) -> None:
        super().__init__()
        
        self.T = n_timesteps 
    
    def forward(
        self, x: torch.Tensor, epsilon: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return NotImplementedError
    
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

class GaussianSchedule(NoiseSchedule):
    def forward(
        self, x: torch.Tensor, epsilon: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        alpha, sigma = self.get_noising_schedulers(t, x)
        z = alpha*x + sigma*epsilon
        
        return z
    
    def alpha_bar(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return NotImplementedError
    
    def get_noising_schedulers(
        self, t: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha_bar = self.alpha_bar(t, x)
        sigma_bar = 1 - alpha_bar
        
        return torch.sqrt(alpha_bar), torch.sqrt(sigma_bar)

    def loss_coef(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class ConstantLinearSchedule(GaussianSchedule):
    def __init__(
        self, n_timesteps: int, beta: Tuple[float, float]=(1e-4, 2e-2)
    ) -> None:
        super().__init__(n_timesteps)

        self.beta = nn.Parameter(
            torch.linspace(beta[0], beta[1], self.T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(
            self.alpha.cumprod(dim=0), requires_grad=False)
    
    def alpha_bar(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self._extract(self.alpha_cumprod, t, x)
    
    def alpha_bar_grid(self, t: torch.Tensor) -> torch.Tensor:
        return self.alpha_cumprod

    def _alpha(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self._extract(self.alpha, t, x)

    def _beta(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self._extract(self.beta, t, x)
    
    def loss_coef(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

class LearnableLinearSchedule(GaussianSchedule):
    """
    TO-DO: Make it work on continuous t to complete equivalence with reference
    implementation.
    """
    def __init__(
        self, n_timesteps: int, monotonic_net: nn.Module|nn.Sequential,
        gamma_limits: Tuple[float, float]=(-12.0, 12.0)
    ) -> None:
        super().__init__(n_timesteps)

        self.gamma_net = monotonic_net
        gamma_min, gamma_max = torch.tensor(gamma_limits, dtype=torch.float)
        self.register_buffer("gamma_min", gamma_min)
        self.register_buffer("gamma_max", gamma_max)

    def alpha_bar(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g = self._gamma(t)
        return self._broadcast(F.sigmoid(-g), x)

    def alpha_bar_grid(self, t: torch.Tensor) -> torch.Tensor:
        return -self._gamma(t)

    def loss_coef(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        s = t - 1
        return self._expm1_gamma(t, s, x)

    def _gamma(self, t: torch.Tensor, atol: float=1e-8) -> torch.Tensor:
        # (B,) long -> (B, 1) float
        tf = self._t_float(t).unsqueeze(-1)
        g = self.gamma_net(tf)
    
        t0 = torch.zeros_like(tf)
        t1 = torch.ones_like(tf)
        g0 = self.gamma_net(t0)
        g1 = self.gamma_net(t1)

        # normalize to [0, 1]
        u = (g - g0) / (g1 - g0 + atol)
        # then scale to [gamma_min, gamma_max]
        g_scaled = self.gamma_min + u * (self.gamma_max - self.gamma_min)

        # (B, 1) -> (B,)
        return g_scaled.squeeze(-1)

    def _expm1_gamma(
        self, t1: torch.Tensor, t2: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        factor = torch.expm1(self._gamma(t1) - self._gamma(t2))
        return self._broadcast(factor, x)
    
    def _t_float(self, t: torch.Tensor) -> torch.Tensor:
        # map [0, T-1] to [0, 1]
        return t.float() / (self.T - 1)

class DiffusionModel(nn.Module):
    def __init__(
        self, base_dist: nn.Module, noise_schedule: nn.Module,
        score_network: nn.Module|nn.Sequential, n_timesteps: int
    ) -> None:
        super().__init__()
        
        self.base = base_dist
        self.noise_sched = noise_schedule
        self.score_net = score_network
        self.T = n_timesteps
    
    def forward(
        self, x: torch.Tensor, y: torch.Tensor|None=None, **kwargs
    ) -> torch.Tensor:
        return self.get_loss(x, y=y, **kwargs)
    
    def get_loss(
        self, x: torch.Tensor, y: torch.Tensor|None=None, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError
    
    def sample(
        self, sample_shape=(1,), n_steps: int=100, y: torch.Tensor|None=None,
        show_path: bool=False, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

class DiffusionEpsilonParam(DiffusionModel):
    def get_loss(
        self, x: torch.Tensor, y: torch.Tensor|None=None, scale: bool=True,
        **kwargs
    ) -> torch.Tensor:
        noise, noise_pred, t = self.noise_prediction(x, y=y, **kwargs)
        mse = (noise_pred - noise).pow(2).view(x.size(0), -1).mean(-1)
        loss = self.noise_sched.loss_coef(t, x).view(x.size(0)) * mse
        
        if scale:
            return self.T/2 * loss.mean()
        else:
            return loss.mean()

    def noise_prediction(
        self, x: torch.Tensor, y: torch.Tensor|None=None, start_time: int=0,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = torch.randint(
            start_time, self.T, (x.size(0),),
            device=x.device, dtype=torch.long
        )
        epsilon = torch.randn_like(x)

        z_t = self.noise_sched(x, epsilon, t)
        epsilon_theta = self.score_net(z_t, t, y=y, **kwargs)
        
        return epsilon, epsilon_theta, t

    def sample(
        self, sample_shape: Tuple[int]|torch.Size=(1,), n_steps: int|None=None,
        start_time: int=0, y: torch.Tensor|None=None,
        show_path: bool=False, **kwargs
    ) -> torch.Tensor:
        n_steps = n_steps if n_steps is not None else self.T
        z_ts = [self.base().sample(sample_shape)]

        for t in reversed(range(start_time, n_steps)):
            z_t = z_ts[-1]
            z_ts.append(self.sampling_step(t, z_t, y=y, **kwargs))
        
        if not show_path:
            return z_ts[-1]
        else:
            return torch.stack(z_ts, dim=0)

    def sampling_step(
        self, t: int, z_t: torch.Tensor, y: torch.Tensor|None=None, **kwargs
    ) -> torch.Tensor:
        return NotImplementedError

class DDPMReverseProcess(DiffusionEpsilonParam):
    def sampling_step(
        self, t: int, z_t: torch.Tensor, y: torch.Tensor|None=None,
        use_beta_tilde: bool=True, **kwargs
    ) -> torch.Tensor:
        # broadcast time to a Tensor of shape (batch_size,)
        t_batch = torch.full(
            (z_t.size(0),), t, device=z_t.device, dtype=torch.long)

        alpha_t = self.noise_sched._alpha(t_batch, z_t)
        alpha_bar_t = self.noise_sched.alpha_bar(t_batch, z_t)
        beta_t = self.noise_sched._beta(t_batch, z_t)

        mean = 1/torch.sqrt(alpha_t) * (
            z_t - beta_t/torch.sqrt(1 - alpha_bar_t)
                * self.score_net(z_t, t_batch, y=y, **kwargs)
        )

        if t > 0:
            if use_beta_tilde:
                alpha_bar_s = self.noise_sched.alpha_bar(t_batch - 1, z_t)
                variance = beta_t * (1 - alpha_bar_s)/(1 - alpha_bar_t)
            else:
                variance = beta_t
            
            return mean + torch.sqrt(variance) * torch.randn_like(z_t)
        else:
            return mean

class VDMReverseProcess(DiffusionEpsilonParam):
    def sampling_step(
        self, t: int, z_t: torch.Tensor, y: torch.Tensor|None=None, **kwargs
    ) -> torch.Tensor:
        # broadcast time to a Tensor of shape (batch_size,)
        t_batch = torch.full(
            (z_t.size(0),), t, device=z_t.device, dtype=torch.long)
        s_batch = t_batch - 1
    
        alpha_bar_s = self.noise_sched.alpha_bar(s_batch, z_t)
        alpha_bar_t = self.noise_sched.alpha_bar(t_batch, z_t)
        r_st = -self.noise_sched._expm1_gamma(s_batch, t_batch, z_t)

        mean = torch.sqrt(alpha_bar_s/alpha_bar_t) * (
            z_t - torch.sqrt(1 - alpha_bar_t) * r_st
                * self.score_net(z_t, t_batch, y=y, **kwargs)
        )

        if t > 0:
            variance = (1 - alpha_bar_s) * r_st
            
            return mean + torch.sqrt(variance) * torch.randn_like(z_t)
        else:
            return mean

class DDIMReverseProcess(DiffusionEpsilonParam):
    def sample(
        self, sample_shape: Tuple[int]|torch.Size=(1,), n_steps: int|None=None,
        start_time: int=0, y: torch.Tensor|None=None, variance: float=1.0,
        spacing: str="logsnr", show_path: bool=False,
        alpha_bar_in_logsnr: bool=False, **kwargs
    ) -> torch.Tensor:
        n_steps = n_steps if n_steps is not None else self.T
        z_ts = [self.base().sample(sample_shape)]
        
        timegrid = torch.linspace(0.0, 1.0, self.T, device=z_ts[-1].device)
        if spacing == "uniform":
            Ts = self._make_ts_uniform(
                self.noise_sched.alpha_bar_grid(timegrid), n_steps, start_time)
        elif spacing == "logsnr":
            Ts = self._make_ts_logsnr(
                self.noise_sched.alpha_bar_grid(timegrid), n_steps, start_time,
                alpha_bar_in_logsnr=alpha_bar_in_logsnr
            )
        for tau_t, tau_s in zip(Ts[:-1], Ts[1:]):
            z_t = z_ts[-1]
            z_ts.append(self.sampling_step(
                start_time, tau_t, tau_s, z_t, y=y, eta=variance, **kwargs))
        
        if not show_path:
            return z_ts[-1]
        else:
            return torch.stack(z_ts, dim=0)
    
    def sampling_step(
        self, start_time: int, t: int, s: torch.Tensor, z_t: torch.Tensor,
        y: torch.Tensor|None=None, eta: float=1.0, **kwargs
    ) -> torch.Tensor:
        # broadcast time to a Tensor of shape (batch_size,)
        t_batch = torch.full(
            (z_t.size(0),), t, device=z_t.device, dtype=torch.long)
        s_batch = torch.full(
            (z_t.size(0),), s, device=z_t.device, dtype=torch.long)

        alpha_bar_s = self.noise_sched.alpha_bar(s_batch, z_t)
        alpha_bar_t = self.noise_sched.alpha_bar(t_batch, z_t)
        variance = (
            eta**2
                * ((1 - alpha_bar_s)/(1 - alpha_bar_t))
                * (1 - alpha_bar_t/alpha_bar_s)
        )

        epsilon_theta = self.score_net(z_t, t_batch, y=y, **kwargs)
        mean = (
            torch.sqrt(alpha_bar_s/alpha_bar_t)
                * (z_t - torch.sqrt(1 - alpha_bar_t) * epsilon_theta)
        )

        if s > start_time:
            direction = torch.sqrt(1 - alpha_bar_s - variance) * epsilon_theta
            return (
                mean + direction + torch.sqrt(variance) * torch.randn_like(z_t))
        else:
            direction = torch.sqrt(1 - alpha_bar_s) * epsilon_theta
            return mean + direction

    @staticmethod
    def _make_ts_uniform(
        alpha_cumprod: torch.Tensor, n_steps: int, start_time: int=0
    ) -> torch.Tensor:
        a = torch.sqrt(alpha_cumprod)
        targets = torch.linspace(a[-1], a[0], steps=n_steps, device=a.device)
        idxes = torch.argmin((a[None, :] - targets[:, None]).abs(), dim=1)

        if idxes[-1] < start_time:
            idxes = idxes[:-1]
        if idxes[-1] != start_time:
            idxes = torch.cat(
                [idxes, torch.tensor([start_time], device=idxes.device)])

        return idxes
    
    @staticmethod
    def _make_ts_logsnr(
        alpha_cumprod: torch.Tensor, n_steps: int, start_time: int=0,
        alpha_bar_in_logsnr: bool=False
    ) -> torch.Tensor:
        if not alpha_bar_in_logsnr:
            a_bar = alpha_cumprod
            log_snr = torch.log(a_bar) - torch.log1p(-a_bar)
        else:
            log_snr = alpha_cumprod
        targets = torch.linspace(
            log_snr[-1], log_snr[0], steps=n_steps,
            device=alpha_cumprod.device
        )

        idxes = torch.argmin(
            (log_snr[None, :] - targets[:, None]).abs(), dim=1)
        idxes = torch.unique_consecutive(idxes.flip(0)).flip(0)
        if idxes[0] != alpha_cumprod.numel() - 1:
            idxes = torch.cat([torch.tensor(
                [alpha_cumprod.numel()-1], device=idxes.device
            ), idxes])

        if idxes[-1] < start_time:
            idxes = idxes[:-1]
        if idxes[-1] != start_time:
            idxes = torch.cat(
                [idxes, torch.tensor([start_time], device=idxes.device)])

        return idxes