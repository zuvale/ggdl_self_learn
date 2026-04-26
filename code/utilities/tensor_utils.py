import torch


def broadcast_(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Broadcast some parameter to data shape.
    """
    return v.view(v.size(0), *([1]*(x.dim() - 1)))