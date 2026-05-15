## TO-DO:
##  - combine the two sinusoidal PE classes into one
import torch
import torch.nn as nn


class DiscreteSinusoidalPE(nn.Module):
    """
    Implementation taken from
    https://medium.com/@hirok4/understanding-transformer-sinusoidal-position-embedding-7cbaaf3b9f6a
    """
    def __init__(
        self, embedding_dim: int, timesteps: int, max_period: float=10000.0
    ) -> None:
        super().__init__()

        time_axis = torch.arange(timesteps).view(timesteps, 1)
        emb_axis = torch.arange(embedding_dim).view(1, embedding_dim)

        angle_rates = 1 / torch.pow(
            torch.tensor(max_period), (2 * (emb_axis // 2)) / embedding_dim)
        angle_rads = time_axis * angle_rates

        pos_enc = torch.zeros_like(angle_rads)
        pos_enc[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        pos_enc[:, 1::2] = torch.cos(angle_rads[:, 1::2])

        self.pos_enc = nn.Parameter(pos_enc, requires_grad=False)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.pos_enc[t]

class ContinuousSinusoidalPE(nn.Module):
    def __init__(
        self, embedding_dim: int, max_period: float=10000.0,
        time_scale: float=1.0
    ) -> None:
        super().__init__()

        emb_axis = torch.arange(embedding_dim).view(1, embedding_dim)
        angle_rates = 1 / torch.pow(
            torch.tensor(max_period), (2 * (emb_axis // 2)) / embedding_dim)
        
        self.register_buffer("angle_rates", angle_rates)
        self.time_scale = time_scale
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float().view(t.size(0), 1)
        angle_rads = self.time_scale * t * self.angle_rates

        pos_enc = torch.zeros_like(angle_rads)
        pos_enc[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        pos_enc[:, 1::2] = torch.cos(angle_rads[:, 1::2])

        return pos_enc

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

class ScalarFourierEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, n_freqs: int=16) -> None:
        super().__init__()
        freqs = 2.0 ** torch.arange(n_freqs).float()
        self.register_buffer("freqs", freqs)

        self.mlp = nn.Sequential(
            nn.Linear(1 + 2 * n_freqs, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch], assumed normalized to roughly [0, 1]
        x = x[:, None]
        angles = 2 * torch.pi * x * self.freqs[None, :]
        feats = torch.cat([x, torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.mlp(feats)

def define_embedding(name: str, **kwargs) -> nn.Module:
    match name:
        case "class":
            return ClassEmbedding(**kwargs)
        case "cont_sine_pe":
            return ContinuousSinusoidalPE(**kwargs)
        case "scalar_fourier":
            return ScalarFourierEmbedding(**kwargs)