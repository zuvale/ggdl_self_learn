import math
import numpy as np
import torch
import torch.nn as nn
from torchmetrics.image.fid import FrechetInceptionDistance
from typing import Dict, Tuple


def mean_flat(x: torch.Tensor) -> torch.Tensor:
    """
    Average non-batch dimensions.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape (B, C, H, W)
    
    Returns
    -------
    torch.Tensor
        Tensor of shape (B,)
    """
    return x.mean(dim=list(range(1, x.dim())))

def normal_kl(
    mean1: torch.Tensor, logvar1: torch.Tensor, mean2: torch.Tensor,
    logvar2: torch.Tensor
) -> torch.Tensor:
    """
    Calculate closed-form KL-divergence between two Gaussians.

    Calculated as KL( N(mean1, exp(logvar1)) || N(mean2, exp(logvar2)) ).
    """
    return 0.5 * (
        - 1.0 + (logvar2 - logvar1) + torch.exp(logvar1 - logvar2)
            + (mean1 - mean2).pow(2) * torch.exp(-logvar2)
    )

def approximate_standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """
    Fast approximation to Phi(x), the standard-normal CDF.
    """
    return 0.5 * (
        1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
        )
    )

def discretized_gaussian_log_likelihood(
    x: torch.Tensor, means: torch.Tensor, log_std: torch.Tensor
) -> torch.Tensor:
    """
    Discretized Gaussian likelihood for images in [-1, 1] representing 8-bit
    values.

    Each pixel is treated as an 8-bit discrete value, and modeled by
    integrating a Gaussian over the bin around that pixel value.
    """
    x_centered = x - means
    inv_std = torch.exp(-log_std)

    # integrate Gaussian over [x - 1/255, x + 1/255]
    plus_intg = inv_std * (x_centered + 1.0 / 255.0)
    minus_intg = inv_std * (x_centered - 1.0 / 255.0)

    cdf_plus = approximate_standard_normal_cdf(plus_intg)
    cdf_minus = approximate_standard_normal_cdf(minus_intg)

    log_cdf_plus = torch.log(torch.clamp(cdf_plus, min=1e-12))
    log_cdf_minus_cdf_minus = torch.log(
        torch.clamp(1.0 - cdf_minus, min=1e-12))

    cdf_delta = cdf_plus - cdf_minus
    log_probs = torch.where(
        x < -0.999, log_cdf_plus, torch.where(
            x > 0.999, log_cdf_minus_cdf_minus,
            torch.log(torch.clamp(cdf_delta, min=1e-12))
        )
    )
    return log_probs

def extract(v: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Extract correct dimensions from time-dependent parameter and broadcast to
    data shape.

    Parameters
    ----------
    v : torch.Tensor
        Vector of shape (T,) containing time-dependent elements
    t : torch.Tensor
        Time steps of shape (B,)
    x : torch.Tensor
        Image tensor of shape (B, C, H, W)
    
    Returns
    -------
    torch.Tensor
        Vector expanded to shape (B, 1, 1, 1) with the correct timepoint
        extracted
    """
    v = v.gather(0, t)
    return v.view(v.size(0), *([1]*(x.dim() - 1)))

class DiffusionBPDEvaluator:
    def __init__(self, diffusion_model: nn.Module) -> None:
        self.diffusion_model = diffusion_model
        self.T = diffusion_model.T
        self.noise_schedule = diffusion_model.noise_sched
        self.score_net = diffusion_model.score_net
    
    @torch.no_grad()
    def _precompute(self, device: str) -> Dict[str, torch.Tensor]:
        """
        Precompute all relevant scalar noising parameters for VLB calculation.

        Calculating from alpha_bar(t).
        """
        # dummy value for broadcasting relevant tensors to correct dimensions
        x_dummy = torch.zeros(self.T, 1, 1, 1, device=device)

        Ts = torch.arange(self.T, device=device, dtype=torch.long)
        alpha_bar = (
            self.noise_schedule
                .alpha_bar(Ts, x_dummy)
                .view(self.T)
                .to(torch.float64)
        )
        # convention: alpha_bar_prev[0] = 1
        alpha_bar_prev = torch.cat([
            torch.ones(1, device=device, dtype=torch.float64), alpha_bar[:-1]])
        # alpha_t / alpha_t-1
        alpha = alpha_bar / alpha_bar_prev
        beta = 1.0 - alpha
        # posterior variance as beta_tilde
        posterior_var = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        post_logvar_clipped = torch.log(
            torch.cat([posterior_var[1:2], posterior_var[1:]]))
        
        noising_mean = beta * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar)
        noising_var = (
            (1.0 - alpha_bar_prev) * torch.sqrt(alpha)/(1.0 - alpha_bar))

        return {
            "alpha_bar": alpha_bar.float(),
            "alpha_bar_prev": alpha_bar_prev.float(), "alpha": alpha.float(),
            "beta": beta.float(), "post_var": posterior_var.float(),
            "post_log_var": post_logvar_clipped.float(),
            "noising_mean": noising_mean.float(),
            "noising_var": noising_var.float(),
            "model_var": posterior_var.float(),
            "model_logvar": post_logvar_clipped.float()
        }
    
    @torch.no_grad()
    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, alpha_bar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample the forward diffusion marginal.

        Essentially runs the forward process. Instead of using the method
        in self.noise_sched, uses a precomputed alpha_bar.
        """
        epsilon = torch.randn_like(x0)
        alpha = extract(alpha_bar.sqrt(), t, x0)
        sigma = extract((1.0 - alpha_bar).sqrt(), t, x0)
        return alpha*x0 + sigma*epsilon, epsilon
    
    @torch.no_grad()
    def p_mean_variance(
        self, x_t: torch.Tensor, t: torch.Tensor,
        cache: Dict[str, torch.Tensor], y: torch.Tensor|None=None,
        clip_denoised: bool=True, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get mean and variance of reverse distribution of noising process.
        """
        alpha_bar = cache["alpha_bar"]
        epsilon_theta = self.score_net(x_t, t, y=y, **kwargs)

        alpha_bar_t = extract(alpha_bar, t, x_t)
        x0_pred = (
            (x_t - torch.sqrt(1.0 - alpha_bar_t) * epsilon_theta)
                / torch.sqrt(alpha_bar_t)
        )
        if clip_denoised:
            x0_pred = x0_pred.clamp(-1.0, 1.0)
        
        model_mean = (
            extract(cache["noising_mean"], t, x_t) * x0_pred
                + extract(cache["noising_var"], t, x_t) * x_t
        )
        model_logvar = extract(cache["model_logvar"], t, x_t)

        return model_mean, model_logvar, x0_pred

    @torch.no_grad()
    def per_t_terms(
        self, x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor,
        cache: Dict[str, torch.Tensor], y: torch.Tensor|None=None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the per-timestep VRB terms in bits-per-dimension.
        """
        # true posterior q(x_{t-1}|x_t, x_0)
        true_mean = (
            extract(cache["noising_mean"], t, x_t) * x0
                + extract(cache["noising_var"], t, x_t) * x_t
        )
        true_logvar = extract(cache["model_logvar"], t, x_t)

        # model reverse distribution p_theta(x_{t-1}|x_t)
        model_mean, model_logvar, x0_pred = self.p_mean_variance(
            x_t, t, cache, y=y, **kwargs)
        
        # compute KL between true posterior and model reverse
        kl = normal_kl(true_mean, true_logvar, model_mean, model_logvar)
        # then normalize to BPD
        kl = mean_flat(kl)/math.log(2.0)

        # decode negative log-likelihood at t==0
        # the log-prob of the original image without noise, as calculated by
        # a discretized Gaussian to account for the dequantization of the
        # images
        decoder_nll = -discretized_gaussian_log_likelihood(
            x0, model_mean, 0.5*model_logvar)
        decoder_nll = mean_flat(decoder_nll)/math.log(2.0)

        return torch.where(t == 0, decoder_nll, kl), x0_pred
    
    @torch.no_grad()
    def prior_term(
        self, x0: torch.Tensor, cache: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Prior term as calculated by KL( q(x_T | x_0) || N(0, I) ).
        """
        t = torch.full(
            (x0.size(0),), self.T - 1, device=x0.device, dtype=torch.long)
        alpha_bar = extract(cache["alpha_bar"], t, x0)

        mean = torch.sqrt(alpha_bar) * x0
        logvar = torch.log(extract(1.0 - cache["alpha_bar"], t, x0))

        zero_tensor = torch.zeros((1,), device=x0.device, dtype=torch.float64)
        kl = normal_kl(mean, logvar, zero_tensor, zero_tensor)
        return mean_flat(kl)/math.log(2.0)

    @torch.no_grad()
    def calculate_bpd(
        self, x0: torch.Tensor, y: torch.Tensor|None=None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate (approximate, bounded) likelihood normalized to bits-per-dim.
        """
        device = x0.device
        cache = self._precompute(device)

        vlb, x0_mse, epsilon_mse = [], [], []
        for t_i in reversed(range(self.T)):
            t = torch.full(
                (x0.size(0),), t_i, device=x0.device, dtype=torch.long)
            
            # sample an x_t ...
            x_t, epsilon = self.q_sample(x0, t, cache["alpha_bar"])
            # ... and compute ELBO at this timestep
            t_i_term, x0_pred = self.per_t_terms(
                x0, x_t, t, cache, y=y, **kwargs)
            vlb.append(t_i_term)
            
            # also store some diagnostics for good measure
            # MSE of predicted x0
            x0_mse.append(mean_flat((x0_pred - x0)**2))
            # MSE of noise prediction
            epsilon_theta = self.score_net(x_t, t, y=y, **kwargs)
            epsilon_mse.append(mean_flat((epsilon_theta - epsilon)**2))

        vlb = torch.stack(vlb, dim=1)
        x0_mse = torch.stack(x0_mse, dim=1)
        epsilon_mse = torch.stack(epsilon_mse, dim=1)

        prior = self.prior_term(x0, cache)
        total_bpd = vlb.sum(dim=1) + prior

        return {
            "total_bpd": total_bpd, "prior_bpd": prior, "per_t_bpd": vlb,
            "x0_mse": x0_mse, "noise_mse": epsilon_mse
        }

class FIDEvaluator:
    def __init__(
        self, feature_net: nn.Module, generative_net: nn.Module,
        n_classes: int, batch_size: int=256, img_size: Tuple[int]=(1, 28, 28)
    ) -> None:
        self.feat_net = feature_net
        self.gen_net = generative_net
        self.n_classes = n_classes
        self.fids = []
        self.n_per_class = None
        self.batch_size = batch_size
        self.img_size = img_size

    def build_real_cache(
        self, real_loader: torch.utils.data.DataLoader, device: str="cpu"
    ) -> None:
        per_class_fids = []
        # instantiate all per-class calculators
        for _ in range(self.n_classes):
            fid = FrechetInceptionDistance(
                feature=self.feat_net, normalize=False,
                input_img_size=self.img_size, reset_real_features=False
            ).to(device)
            fid.set_dtype(torch.float64)
            per_class_fids.append(fid)
        
        per_class_nums = np.zeros((self.n_classes,)).astype(int)
        for x, y in real_loader:
            y = y.to(device)
            for c in range(self.n_classes):
                cond = (y == c)
                mask = cond
                if mask.any():
                    per_class_fids[c].update(x[mask.to(x.device)], real=True)
                    per_class_nums[c] += cond.sum()
        
        self.fids = per_class_fids
        self.n_per_class = per_class_nums
    
    def calculate_conditional_fid(
        self, device: str="cpu", class_multiplier: int=1, **kwargs
    ) -> torch.Tensor:
        scores = []
        for c in range(self.n_classes):
            fid = self.fids[c]
            fid.reset()

            remaining = self.n_per_class[c] * class_multiplier
            while remaining > 0:
                batch_size = min(remaining, self.batch_size)
                y = torch.full(
                    (batch_size,), c, device=device, dtype=torch.long)
                x_gen = self.gen_net.sample((batch_size,), y=y, **kwargs)
                x_gen = x_gen/2 + 0.5

                fid.update(x_gen, real=False)
                remaining -= batch_size
            
            scores.append(fid.compute())
        
        return torch.stack(scores)