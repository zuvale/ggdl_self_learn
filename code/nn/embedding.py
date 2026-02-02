import torch
import torch.nn as nn


class SinusoidalPE(nn.Module):
    """
    Implementation taken from
    https://medium.com/@hirok4/understanding-transformer-sinusoidal-position-embedding-7cbaaf3b9f6a
    """
    def __init__(self, embedding_dim: int, timesteps: int) -> None:
        super().__init__()

        time_axis = torch.arange(timesteps).view(timesteps, 1)
        emb_axis = torch.arange(embedding_dim).view(1, embedding_dim)

        angle_rates = 1 / torch.pow(10000, (2 * (emb_axis // 2)) / embedding_dim)
        angle_rads = time_axis * angle_rates

        pos_enc = torch.zeros_like(angle_rads)
        pos_enc[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        pos_enc[:, 1::2] = torch.cos(angle_rads[:, 1::2])

        self.pos_enc = nn.Parameter(pos_enc, requires_grad=False)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.pos_enc[t]

class ClassEmbedding(nn.Module):
    def __init__(
        self, embedding_dim: int, n_classes: int
    ) -> None:
        super().__init__()

        self.enc = nn.Parameter(torch.randn(n_classes, embedding_dim)*1e-2)
    
    def forward(
        self, y: torch.Tensor|None=None) -> torch.Tensor:
        if y is not None:
            # if some classes are missing (e.g. for unconditional training
            # with dropout)
            if (y < 0).any():
                m = (y < 0).float().view(-1, 1, 1)
                c = (
                    (1 - m) * self.enc[y]
                        + m * self.enc.mean(dim=0).unsqueeze(0)
                )
            else:
                c = self.enc[y]
        else:
            c = self.enc.mean(dim=0).unsqueeze(0)
        
        return c