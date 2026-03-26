import torch
import torch.nn as nn
import torch.distributions as td
from torch.distributions.kl import register_kl
from typing import Tuple

from .utils import create_linear_schedule


class Encoder(nn.Module):
    def __init__(self, encoder_net: nn.Module|nn.Sequential) -> None:
        super().__init__()

        self.net = encoder_net
    
    def forward(
        self, x: torch.Tensor, y: torch.Tensor|None=None) -> td.Distribution:
        x = self.net(x, y=y)
        return self.reparameterize(x)
    
    def reparameterize(self, x: torch.Tensor) -> td.Distribution:
        raise NotImplementedError

class GaussianEncoder(Encoder):    
    def reparameterize(self, x: torch.Tensor) -> td.Distribution:
        mean, logvar = torch.chunk(x, 2, dim=-1)
        return td.Independent(
            td.Normal(loc=mean, scale=torch.exp(0.5 * logvar)), 1)

class Decoder(nn.Module):
    def __init__(
        self, decoder_net: nn.Module|nn.Sequential, dimensionality: Tuple
    ) -> None:
        super().__init__()

        self.net = decoder_net
        self.dim = dimensionality
        if isinstance(self.dim, (tuple, list, torch.Size)):
            self.event_ndims = len(self.dim)
        else:
            self.event_ndims = 1
    
    def forward(
        self, x: torch.Tensor, y: torch.Tensor|None=None) -> td.Distribution:
        x = self.net(x, y=y)
        return self.out_dist(x)
    
    def out_dist(self, x: torch.Tensor) -> td.Distribution:
        raise NotImplementedError

class BernoulliDecoder(Decoder):    
    def out_dist(self, x: torch.Tensor) -> td.Distribution:
        return td.Independent(td.Bernoulli(logits=x), self.event_ndims)

class GaussianDecoder(Decoder):
    def __init__(
        self, decoder_net: nn.Module|nn.Sequential, dimensionality: Tuple,
        variance_type: str="scalar", logvar_init: float=-4.6
    ) -> None:
        super().__init__(
            decoder_net=decoder_net, dimensionality=dimensionality)

        match variance_type:
            case "scalar":
                self.logvar = nn.Parameter(
                    torch.ones((1,)) * logvar_init, requires_grad=True)
            case "diagonal":
                self.logvar = nn.Parameter(
                    torch.ones(*self.dim) * logvar_init,
                    requires_grad=True
                )

    def out_dist(self, x: torch.Tensor) -> td.Distribution:
        means = x
        return td.Independent(
            td.Normal(loc=means, scale=torch.exp(0.5 * self.logvar)),
            self.event_ndims
        )

class VAE(nn.Module):
    def __init__(
            self, prior: nn.Module, encoder: nn.Module|nn.Sequential,
            decoder: nn.Module|nn.Sequential
        ) -> None:
        super().__init__()

        self.prior = prior
        self.enc = encoder
        self.dec = decoder
    
    def forward(
            self, x: torch.Tensor, y: torch.Tensor|None=None
    ) -> tuple[td.Distribution, td.Distribution]:
        posterior = self.enc(x, y=y)
        latent = posterior.rsample()
        lik = self.dec(latent, y=y)
        return posterior, lik

    def sample(
        self, sample_shape: Tuple[int]|torch.Size=(1,),
        y: torch.Tensor|None=None, use_mean: bool=False, **kwargs
    ) -> torch.Tensor:
        latent = self._prior_prep().sample(sample_shape)
        likelihood = self.dec(latent, y=y, **kwargs)
        if not use_mean:
            return likelihood.sample()
        else:
            return likelihood.mean
    
    def _prior_prep(self) -> td.Distribution:
        return self.prior()

class ELBO(nn.modules.loss._Loss):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(
            self, prior: td.Distribution|nn.Module, posterior: td.Distribution,
            likelihood: td.Distribution, target: torch.Tensor, *args
        ) -> torch.Tensor:
        reconstruction = self.recon_loss(likelihood, target)
        regularization = self.regul_loss(posterior, prior)

        elbo = torch.mean(reconstruction - regularization, dim=0)
        return -elbo, (reconstruction, regularization)
    
    def recon_loss(
        self, likelihood: td.Distribution, target: torch.Tensor
    ) -> torch.Tensor:
        return likelihood.log_prob(target)
    
    def regul_loss(
        self, posterior: td.Distribution, prior: td.Distribution
    ) -> torch.Tensor:
        return td.kl_divergence(posterior, prior)

class ELBOAnnealedBeta(ELBO):
    def __init__(
        self, total_steps: int, beta_start: float, beta_end: float=1.0,
        **kwargs
    ) -> None:
        super().__init__()

        self.betas = create_linear_schedule(
            total_steps, beta_start, beta_end, **kwargs)
    
    def forward(
        self, prior: td.Distribution|nn.Module, posterior: td.Distribution,
        likelihood: td.Distribution, target: torch.Tensor, step: int
    ) -> torch.Tensor:
        reconstruction = self.recon_loss(likelihood, target)
        regularization = self.regul_loss(posterior, prior)

        beta = self.betas[step]
        elbo = torch.mean(reconstruction - beta * regularization, dim=0)
        return -elbo, (reconstruction, beta * regularization)

@register_kl(td.Independent, td.MixtureSameFamily)
def _independentnormal_mixturesamefamily_approximate(p, q, n_samples=5000):
    samples = p.rsample((n_samples,))
    log_p = p.log_prob(samples)
    
    log_q = q.log_prob(samples)

    return (log_p - log_q).mean(dim=0)