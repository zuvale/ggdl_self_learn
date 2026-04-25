import torch
import torch.nn as nn
from typing import Tuple


def explicit_euler_fixed(
    fun: nn.Module, t_eval: Tuple[int|float, int|float], X0: torch.Tensor,
    N: int, device: str="cpu", args=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if args is not None:
        fun = lambda x, t, fun=fun: fun(x, t, *args)
    
    t0, t_end = t_eval
    h = (t_end - t0)/N

    T = torch.zeros((N+1, X0.shape[0], 1), device=device)
    T[0] = t0
    Y = torch.zeros((N+1, *X0.shape), device=device)
    Y[0] = X0

    for k in range(N):
        T[k+1] = T[k] + h
        Y[k+1] = Y[k] + h*fun(Y[k], T[k])
    
    return Y, T

def midpoint_fixed(
    fun: nn.Module, t_eval: Tuple[int|float, int|float], X0: torch.Tensor,
    N: int, device: str="cpu", args=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if args is not None:
        fun = lambda x, t, fun=fun: fun(x, t, *args)
    
    t0, t_end = t_eval
    h = (t_end - t0)/N

    T = torch.zeros((N+1, X0.shape[0], 1), device=device)
    T[0] = t0
    Y = torch.zeros((N+1, *X0.shape), device=device)
    Y[0] = X0
    
    for k in range(N):
        T[k+1] = T[k] + h
        Y[k+1] = Y[k] + h*fun(
            Y[k] + (h/2) * fun(Y[k], T[k]), T[k] + h/2
        )
    
    return Y, T