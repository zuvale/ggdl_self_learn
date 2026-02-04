from torch import Tensor
import torch.nn as nn

class DummyLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return x