## TO-DO:
# - reduce boilerplate in NFPatchified class
import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F
from typing import Iterable
from nn_.transformer import CausalSDPA


def checkerboard_mask_1d(
    l: int, invert: bool=False, device: str=None) -> torch.Tensor:
    m = torch.zeros(l)
    m[:l//2] = 1.0
    if invert:
        m = 1 - m
    return m.view(1, l).to(device)

def checkerboard_mask_2d(
    h: int, w: int, invert: bool=False, device: str=None) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    m = (yy + xx) % 2
    if invert:
        m = 1 - m
    return m.to(torch.float32).view(1, 1, h, w).to(device)

def channel_mask(c: int, invert: bool=False, device: str=None) -> torch.Tensor:
    m = torch.zeros(c)
    m[:c//2] = 1.0
    if invert:
        m = 1 - m
    return m.view(1, c, 1, 1).to(device)

class Permutation(nn.Module):
    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len
    
    def forward(
        self, x: torch.Tensor, dim: int=1, inverse: bool=False
    ) -> torch.Tensor:
        raise NotImplementedError

class PermutationIdentity(Permutation):
    def forward(
        self, x: torch.Tensor, dim: int=1, inverse: bool=False
    ) -> torch.Tensor:
        return x

class PermutationFlip(Permutation):
    def forward(
        self, x: torch.Tensor, dim: int=1, inverse: bool=False
    ) -> torch.Tensor:
        return x.flip(dims=[dim])

class ActNorm1d(nn.Module):
    """
    Invertible affine normalization for (B, F).
    Data-dependent initialization on first forward pass.
    """
    def __init__(self, n_features: int, eps: float=1e-6) -> None:
        super().__init__()

        self.eps = eps
        self.initialized = False
        
        self.bias = nn.Parameter(torch.zeros(1, n_features))
        self.log_scale = nn.Parameter(torch.zeros(1, n_features))

    @torch.no_grad()
    def _data_init(self, x: torch.Tensor) -> None:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp_min(self.eps)

        self.bias.data = -mean
        self.log_scale.data = torch.log(1.0 / std)
        self.initialized = True
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized:
            self._data_init(x)
        
        z = (x + self.bias) * torch.exp(self.log_scale)
        log_det_J = self.log_scale.sum(dim=1).expand(x.size(0))
        
        return z, log_det_J
    
    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = z * torch.exp(-self.log_scale) - self.bias
        log_det_J = -self.log_scale.sum(dim=1).expand(z.size(0))

        return x, log_det_J

class ActNorm2d(nn.Module):
    """
    Invertible per-channel affine normalization for (B, C, H, W).
    Data-dependent initialization on first forward pass.
    """
    def __init__(self, n_channels: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.eps = eps
        self.initialized = False

        self.bias = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, n_channels, 1, 1))

    @torch.no_grad()
    def _data_init(self, x: torch.Tensor) -> None:
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        std = x.std(dim=(0, 2, 3), keepdim=True).clamp_min(self.eps)

        self.bias.data = -mean
        self.log_scale.data = torch.log(1.0 / std)
        self.initialized = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized:
            self._data_init(x)

        z = (x + self.bias) * torch.exp(self.log_scale)

        _, C, H, W = x.size()
        logdet = (
            self.log_scale.view(1, C).sum(dim=1) * (H * W)).expand(x.size(0))

        return z, logdet

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = z * torch.exp(-self.log_scale) - self.bias

        _, C, H, W = z.size()
        logdet = (
            -self.log_scale.view(1, C).sum(dim=1) * (H * W)).expand(z.size(0))
        return x, logdet

class NormalizingFlow(nn.Module):
    def __init__(
        self, base_dist: nn.Module, trafo_blocks: nn.ModuleList,
        device: str="cpu"
    ) -> None:
        super().__init__()

        self.device = device
        self.base = base_dist.to(device=self.device)
        self.transformations = trafo_blocks
    
    def forward(
        self, z: torch.Tensor, y: torch.Tensor|None=None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.reverse_norm(z, y=y, **kwargs)
    
    def reverse_norm(
        self, z: torch.Tensor, y: torch.Tensor|None=None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_dets = torch.zeros(z.size(0), device=z.device)
        for T in self.transformations:
            z, log_det_J = T.reverse_norm(z, y=y, **kwargs)
            log_dets = log_dets + log_det_J
        
        return z, log_dets
    
    @torch.inference_mode()
    def forward_flow(
        self, sample_shape: Tuple=(1,), y: torch.Tensor|None=None,
        show_path: bool=False, **kwargs
    ) -> torch.Tensor:
        x = self.base().sample(sample_shape)
        if show_path:
            xs = [x]

        for T in reversed(self.transformations):
            x = T.forward_flow(x, y=y, **kwargs)
            if show_path:
                xs.append(x)

        if not show_path:
            return x
        else:
            return torch.stack(xs, dim=0)

class NFPatchified(NormalizingFlow):
    def __init__(
        self, base_dist: nn.Module, trafo_blocks: nn.ModuleList,
        patch_size: int, data_size: int|Tuple[int, int], device: str="cpu"
    ) -> None:
        super().__init__(base_dist, trafo_blocks, device=device)

        self.patch_size = patch_size
        if isinstance(data_size, int):
            self.data_size = (data_size, data_size)
        else:
            self.data_size = data_size
    
    def forward(
        self, z: torch.Tensor, y: torch.Tensor|None=None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.reverse_norm(z, y=y, **kwargs)

    def reverse_norm(
        self, z: torch.Tensor, y: torch.Tensor|None=None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self._patchify(z)
        log_dets = torch.zeros(z.size(0), device=z.device)
        for T in self.transformations:
            z, log_det_J = T.reverse_norm(z, y=y, **kwargs)
            log_dets = log_dets + log_det_J
        
        return z, log_dets

    @torch.inference_mode()
    def forward_flow(
        self, sample_shape: Tuple=(1,), u: torch.Tensor|None=None,
        y: torch.Tensor|None=None, show_path: bool=False, **kwargs
    ) -> torch.Tensor:
        if u is None:
            u = self.base().sample(sample_shape)

        if show_path:
            us = [u]

        for T in reversed(self.transformations):
            u = T.forward_flow(u, y=y, **kwargs)
            if show_path:
                us.append(u)

        if not show_path:
            return self._unpatchify(u)
        else:
            return torch.stack([self._unpatchify(u) for u in us], dim=0)
    
    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, no_of_channels, height, width)
        # -> (batch_size, token_size, no_of_patch_channels)
        u = F.unfold(x, self.patch_size, stride=self.patch_size)
        return u.transpose(1, 2)
    
    def _unpatchify(self, u: torch.Tensor) -> torch.Tensor:
        # (batch_size, token_size, no_of_patch_channels)
        # -> (batch_size, no_of_channels, height, width)
        x = F.fold(
            u.transpose(1, 2), self.data_size, self.patch_size,
            stride=self.patch_size
        )
        return x

class RealNVPCoupling(nn.Module):
    def __init__(
        self, coupling_net: nn.Module|nn.Sequential, mask: torch.Tensor,
        clamp_factor: float=5.0
    ) -> None:
        super().__init__()

        self.coupling_net = coupling_net
        self.register_buffer("mask", mask)

        self.clamp_factor = clamp_factor
    
    def forward(
        self, z: torch.Tensor, y: torch.Tensor|None=None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.reverse_norm(z, y=y, **kwargs)
    
    def reverse_norm(
        self, z: torch.Tensor, y: torch.Tensor|None=None,
        clamp_factor: float=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if clamp_factor is None:
            clamp_factor = self.clamp_factor

        # (batch_size, 2*sequence_length)
        z_a = z * self.mask

        log_scale, shift = self.coupling_net(z_a, y=y)
        log_scale = self._clamp(
            log_scale * (1 - self.mask), clamp_factor=clamp_factor)
        shift = shift * (1 - self.mask)
        scale = torch.exp(-log_scale.float()).type(log_scale.dtype)

        z = z_a + (1 - self.mask) * ((z - shift) * scale)
        log_det_J = (-(log_scale)).view(z.size(0), -1).sum(dim=1)

        return z, log_det_J

    def forward_flow(
        self, x: torch.Tensor, y: torch.Tensor|None=None,
        clamp_factor: float=None, guidance: float=0, guide_what: str="ab"
    ) -> torch.Tensor:
        """
        In guide_what:
         - a corresponds to (log) scale (exponent)
         - b corresponds to shift
        """
        if clamp_factor is None:
            clamp_factor = self.clamp_factor

        # (batch_size, sequence_length)
        x_a = x * self.mask

        log_scale, shift = self.coupling_net(x_a, y=y)

        if guidance > 0 and guide_what:
            log_scale_u, shift_u = self.coupling_net(x_a, y=None)
            g = guidance
            
            if "a" in guide_what:
                log_scale = log_scale + g * (log_scale - log_scale_u)
            if "b" in guide_what:
                shift = shift + g * (shift - shift_u)
        
        log_scale = self._clamp(
            log_scale * (1 - self.mask), clamp_factor=clamp_factor)
        shift = shift * (1 - self.mask)
        scale = torch.exp(log_scale.float()).type(log_scale.dtype)

        x = x_a + (1 - self.mask) * (x * scale + shift)
        return x
    
    @staticmethod
    def _clamp(log_scale: torch.Tensor, clamp_factor: float) -> torch.Tensor:
        return clamp_factor * torch.tanh(log_scale/clamp_factor)

class RealNVPBlock(nn.Module):
    def __init__(
        self, coupling_nets: nn.ModuleList, norms: nn.ModuleList,
        masks: Iterable[torch.Tensor]|torch.Tensor, squeeze: bool=True,
        **kwargs
    ) -> None:
        assert len(coupling_nets) == len(masks) == len(norms)
        super().__init__()

        self.couplings = nn.ModuleList([
            RealNVPCoupling(c_net, m, **kwargs)
            for c_net, m in zip(coupling_nets, masks)
        ])
        self.norms = norms
        self.squeeze_flag = squeeze
    
    def forward(
        self, z: torch.Tensor, y: torch.Tensor|None=None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.reverse_norm(z, y=y, **kwargs)

    def reverse_norm(
        self, z: torch.Tensor, y: torch.Tensor|None=None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_dets = torch.zeros(z.size(0), device=z.device)
        for coup, norm in zip(self.couplings, self.norms):
            z, log_det_J_coup = coup.reverse_norm(z, y=y, **kwargs)
            z, log_det_J_norm = norm(z)
            log_dets = log_dets + log_det_J_coup + log_det_J_norm
        
        if self.squeeze_flag:
            z = self._squeeze2x(z)
        
        return z, log_dets

    def forward_flow(
        self, x: torch.Tensor, y: torch.Tensor|None=None, **kwargs
    ) -> torch.Tensor:
        if self.squeeze_flag:
            x = self._unsqueeze2x(x)

        for coup, norm in zip(reversed(self.couplings), reversed(self.norms)):
            x, _ = norm.inverse(x)
            x = coup.forward_flow(x, y=y, **kwargs)
        
        return x

    def _squeeze2x(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        x = x.view(B, C, H//2, 2, W//2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(B, C*4, H//2, W//2)
    
    def _unsqueeze2x(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        x = x.view(B, C//4, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        return x.view(B, C//4, H*2, W*2)

class TarFlowBlock(nn.Module):
    """
    TO-DO: Add the volume-preserving mode as an option
    """
    def __init__(
        self,
        permutation: Permutation, attention_block: nn.ModuleList,
        in_chans: int, hidden_chans: int, n_patches: int, n_classes: int=0
    ) -> None:
        super().__init__()
        
        self.perm = permutation
        self.attn_blocks = attention_block
        self.proj_in = nn.Linear(in_chans, hidden_chans)
        self.pos_embed = nn.Parameter(
            torch.randn(n_patches, hidden_chans)*1e-2)
        if n_classes > 0:
            self.class_embed = nn.Parameter(
                torch.randn(n_classes, 1, hidden_chans)*1e-2)
        else:
            self.class_embed = None
        
        self.proj_out = nn.Linear(hidden_chans, in_chans*2)
        self.proj_out.weight.data.fill_(0.0)

        self.register_buffer(
            "attn_mask",
            torch.tril(torch.ones(n_patches, n_patches, dtype=torch.bool))
        )
    
    def forward(
        self, z: torch.Tensor, y: torch.Tensor|None=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.reverse_norm(z, y=y)

    def reverse_norm(
        self, z: torch.Tensor, y: torch.Tensor|None=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # (batch_size, no_of_tokens, input_channels)
        z = self.perm(z)
        z_tilde = z
        # (batch_size, no_of_tokens, hidden_channels)
        z = self.proj_in(z) + self.perm(self.pos_embed, dim=0)
        if self.class_embed is not None:
            if y is not None:
                # if some classes are missing (e.g. for unconditional training
                # with dropout)
                if (y < 0).any():
                    m = (y < 0).float().view(-1, 1, 1)
                    class_embed = (
                        (1 - m) * self.class_embed[y]
                            + m * self.class_embed.mean(dim=0)
                    )
                else:
                    class_embed = self.class_embed[y]
                z = z + class_embed
            else:
                z = z + self.class_embed.mean(dim=0)

        for block in self.attn_blocks:
            z = block(z, self.attn_mask)
        # (batch_size, no_of_tokens, in_channels*2)
        z = self.proj_out(z)
        z = torch.cat([torch.zeros_like(z[:, :1]), z[:, :-1]], dim=1)
        # 2 * (batch_size, no_of_tokens, in_channels)
        log_scale, shift = z.chunk(2, dim=-1)
        scale = torch.exp(-log_scale.float()).type(log_scale.dtype)

        z = (z_tilde - shift) * scale
        log_det_J = -log_scale.mean(dim=(1, 2))

        return self.perm(z, inverse=True), log_det_J
    
    def forward_flow(
        self, x: torch.Tensor, y: torch.Tensor|None=None,
        caching: bool=False, guidance: float=0, guide_what: str="ab",
        attn_temp: float=1.0, annealed_guidance: bool=False
    ) -> torch.Tensor:
        """
        In guide_what:
         - a corresponds to (log) scale (exponent)
         - b corresponds to shift
        """
        self._set_sampling_mode(True)
        B, T, C = x.size()

        x = self.perm(x)
        pos_embed = self.perm(self.pos_embed, dim=0)
        for i in range(T - 1):
            log_scale, shift = self.flow_step(
                x, pos_embed, i, y=y, attn_temp=attn_temp,
                which_cache="cond" if caching else None
            )
            if guidance > 0 and guide_what:
                log_scale_u, shift_u = self.flow_step(
                    x, pos_embed, i, None, attn_temp=attn_temp,
                    which_cache="uncond" if caching else None
                )
                if annealed_guidance:
                    g = (i + 1)/(T - 1) * guidance
                else:
                    g = guidance
                
                if "a" in guide_what:
                    log_scale = log_scale_u + g * (log_scale - log_scale_u)
                if "b" in guide_what:
                    shift = shift_u + g * (shift - shift_u)
            
            scale = torch.exp(log_scale.float()).type(log_scale.dtype)
            x[:, i+1:i+2] = x[:, i+1:i+2]*scale + shift
        
        self._set_sampling_mode(False)
        return self.perm(x, inverse=True)
    
    def flow_step(
        self, x: torch.Tensor, pos_embed: torch.Tensor, i: int,
        y: torch.Tensor|None=None, attn_temp: float=1.0,
        which_cache: str|None=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TO-DO: simplify/reduce boilerplate
        """
        
        # get i-th token/patch
        if which_cache is None:
            # (batch_size, no_of_gen_tokens, in_channels)
            x_in = x[:, :i+1]
            # (batch_size, no_of_gen_tokens, hidden_channels)
            h = self.proj_in(x_in) + pos_embed[:i+1]
            if self.class_embed is not None:
                if y is not None:
                    h = h + self.class_embed[y]
                else:
                    h = h + self.class_embed.mean(dim=0)
            
            mask = self.attn_mask[:i+1, :i+1]
            for block in self.attn_blocks:
                h = block(h, attn_mask=mask, attn_temp=attn_temp)
            # (batch_size, 1, in_channels*2)
            out = self.proj_out(h[:, -1:, :])
        else:
            # (batch_size, 1, in_channels)
            x_in = x[:, i:i+1]
            # (batch_size, 1, hidden_channels)
            x = self.proj_in(x_in) + pos_embed[i:i+1]
            if self.class_embed is not None:
                if y is not None:
                    x = x + self.class_embed[y]
                else:
                    x = x + self.class_embed.mean(dim=0)
            
            for block in self.attn_blocks:
                x = block(x, attn_temp=attn_temp, which_cache=which_cache)
            # (batch_size, 1, in_channels*2)
            out = self.proj_out(x[:, -1:, :])
        
        # 2 * (batch_size, 1, in_channels)
        log_scale, shift = out.chunk(2, dim=-1)
        return log_scale, shift

    def _set_sampling_mode(self, flag: bool) -> None:
        """
        TO-DO: make work for arbitrary attention
        """
        for m in self.modules():
            if isinstance(m, CausalSDPA):
                m.sample = flag
                m.k_cache = {"cond": [], "uncond": []}
                m.v_cache = {"cond": [], "uncond": []}