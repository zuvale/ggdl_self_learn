import torch
import torch.nn as nn
import torch.distributions as td
from typing import Tuple


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
        logit_init: float=3.0, logvar_init: float=-4.6
    ) -> None:
        super().__init__()

        self.dim = dimensionality
        self.k = num_components
        self.mix_logits = nn.Parameter(
            torch.ones(self.k) * logit_init, requires_grad=True)
        self.means = nn.Parameter(
            torch.randn(self.k, *self.dim), requires_grad=True)
        self.logvars = nn.Parameter(torch.ones(self.k, *self.dim) * logvar_init)

    def forward(self) -> td.Distribution:
        mix_dist = td.Categorical(logits=self.mix_logits)
        comp_dist = td.Independent(
            td.Normal(loc=self.means, scale=torch.exp(0.5 * self.logvars)),
            reinterpreted_batch_ndims=1
        )
        return td.MixtureSameFamily(mix_dist, comp_dist)

class VampBase(nn.Module):
    def __init__(
            self, dimensonality: Tuple[int], num_components: int,
            encoder: nn.Module|nn.Sequential, logit_init: float=3.0,
            pseudo_init: float=1.0, unflatten_flag=False
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
    
    def forward(self) -> td.Distribution:
        mix_dist = td.Categorical(logits=self.mix_logits)
        comp_dist = self.enc(self.pseudo_inputs)
        
        return td.MixtureSameFamily(mix_dist, comp_dist)