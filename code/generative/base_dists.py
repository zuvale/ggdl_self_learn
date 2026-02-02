import torch
import torch.nn as nn
import torch.distributions as td
from typing import Tuple


class GaussianBase(nn.Module):
    normalization_constant = 0.5 * torch.log(
        torch.Tensor([2 * torch.pi]))
    def __init__(self, dimensionality: Tuple) -> None:
        super().__init__()

        self.dim = dimensionality
        self.mean = nn.Parameter(torch.zeros(*self.dim), requires_grad=False)
        self.std = nn.Parameter(torch.ones(*self.dim), requires_grad=False)
    
    def forward(self) -> td.Distribution:
        return td.Independent(td.Normal(self.mean, self.std), len(self.dim))