import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from typing import Tuple


def tweedie_denoise(
    model: nn.Module|nn.Sequential, x_noisy: torch.Tensor, y: torch.Tensor,
    sigma: float, clamp: Tuple[float, float]|None=(-1.0, 1.0),
    device: str="cpu"
) -> torch.Tensor:
    x_noisy = x_noisy.detach().requires_grad_(True).to(device)

    z, log_dets = model(x_noisy, y=y)
    log_p = model.base().log_prob(z) + log_dets
    score = torch.autograd.grad(log_p.sum(), x_noisy, create_graph=False)[0]

    x = x_noisy + sigma**2 * score
    if clamp is not None:
        x = x.clamp(clamp[0], clamp[1])
    
    return x.detach()

class SoftPlusParameterization(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

class LinearMonotonicNetwork(nn.Module):
    def __init__(
        self, n_feats: int, hidden_size: int, act_fun: nn.Module=nn.Sigmoid,
        embedding: nn.Module|None=None
    ) -> None:
        super().__init__()
    
        self.layer_1 = nn.Linear(n_feats, n_feats)
        self.layer_2 = nn.Linear(n_feats, hidden_size)
        self.layer_3 = nn.Linear(hidden_size, n_feats)
        for layer in (self.layer_1, self.layer_2, self.layer_3):
            parametrize.register_parametrization(
                layer, "weight", SoftPlusParameterization())
        self.act_fun = act_fun()
        if embedding:
            self.emb = embedding
        else:
            self.emb = None
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if self.emb:
            t = self.emb(t)
        h = self.layer_1(t)
        h_tilde = h
        
        h = self.layer_2(h)
        h = self.act_fun(self.layer_3(h))
        h = h_tilde + h
        
        return h