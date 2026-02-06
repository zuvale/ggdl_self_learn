import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class CausalSDPA(nn.Module):
    def __init__(self, n_channels: int, head_channels: int) -> None:
        assert n_channels % head_channels == 0
        super().__init__()

        self.norm = nn.LayerNorm(n_channels)
        self.qkv = nn.Linear(n_channels, n_channels*3)
        self.proj = nn.Linear(n_channels, n_channels)
        self.n_heads = n_channels // head_channels
        self.sqrt_scale = head_channels**(-0.25)

        self.sample = False
        self.k_cache: Dict[str, List[torch.Tensor]] = {"cond": [], "uncond": []}
        self.v_cache: Dict[str, List[torch.Tensor]] = {"cond": [], "uncond": []}
    
    def forward(
        self, x: torch.Tensor, mask: torch.Tensor|None=None, temp: float=1.0,
        which_cache: None|str=None
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = (
            # (batch_size, no_of_tokens, no_of_channels*3)
            self.qkv(x)
                # (batch_size, no_of_tokens, no_of_heads*3, head_dimensionality)
                .reshape(B, T, 3*self.n_heads, -1)
                # (batch_size, no_of_heads*3, no_of_tokens, head_dimensionality)
                .transpose(1, 2).contiguous()
                # 3 * (batch_size, no_of_heads, no_of_tokens, head_dimensionality)
                .chunk(3, dim=1)
        )

        if self.sample and which_cache is not None:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=2)
            v = torch.cat(self.v_cache[which_cache], dim=2)

        scale = self.sqrt_scale**2 / temp
        if mask is not None:
            mask = mask.bool()

        x = (
            # (batch_size, no_of_heads, no_of_tokens, head_dimensionality)
            F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
                # (batch_size, no_of_tokens, no_of_heads, head_dimensionality)
                .transpose(1, 2)
                # (batch_size, no_of_tokens, no_of_channels)
                .reshape(B, T, C)
        )
        return self.proj(x)