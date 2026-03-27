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