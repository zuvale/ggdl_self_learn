## TO-DO:
# - add shape checks for intermediate steps too in UNet class
# - reduce boilerplate regarding similar structure of CNN2DToFC and UNet
# - move code calculating the convolution output sizes to dedicated functions
#   in another script
import torch
import torch.nn as nn
from typing import List
from .utils import DummyLayer


class Conv2DLayer(nn.Module):
    def __init__(
        self, in_chans: int, out_chans: int, conv_kernel_size: int,
        conv_pad_size: int, conv_stride_size: int=1, pool: None|str="max",
        pool_kernel_size: None|int=None, pool_stride: None|int=None,
        pool_padding: None|int=None,
        skip: bool=False, use_batchnorm: bool=False,
        act_fun: nn.Module=nn.ReLU, act_fun_pos: str="after_pool"
    ) -> None:
        super().__init__()

        modules = []
        modules.append(nn.Conv2d(
            in_chans, out_chans, kernel_size=conv_kernel_size,
            padding=conv_pad_size, stride=conv_stride_size
        ))

        if act_fun is not None and act_fun_pos == "after_conv":
            modules.append(act_fun())

        if pool:
            if pool == "max":
                modules.append(nn.MaxPool2d(
                    pool_kernel_size, stride=pool_stride, padding=pool_padding))
            elif pool == "mean":
                modules.append(nn.AvgPool2d(
                    pool_kernel_size, stride=pool_stride, padding=pool_padding))
        
        if use_batchnorm:
            modules.append(nn.BatchNorm2d(out_chans))

        if act_fun is not None and act_fun_pos == "after_pool":
            modules.append(act_fun())
        
        self.conv2d_layer = nn.Sequential(*modules)

        self.skip_flag = skip
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x_out = self.conv2d_layer(x)
        if self.skip_flag:
            return x + x_out
        else:
            return x_out

class TransposeConv2DLayer(nn.Module):
    def __init__(
        self, in_chans: int, out_chans: int, kernel_size: int, in_pad_size: int,
        out_pad_size: int, stride_size: int, use_batchnorm: bool=False,
        act_fun: nn.Module=nn.ReLU
    ) -> None:
        super().__init__()
        
        modules = []
        modules.append(nn.ConvTranspose2d(
            in_chans, out_chans, kernel_size=kernel_size, stride=stride_size,
            padding=in_pad_size, output_padding=out_pad_size
        ))

        if use_batchnorm:
            modules.append(nn.BatchNorm2d(out_chans))
        if act_fun is not None:
            modules.append(act_fun())
        
        self.tconv2d_layer = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tconv2d_layer(x)

class CNN2DToFC(nn.Module):
    def __init__(
        self, n_feats: int, in_channels: int, out_channels: int,
        hidden_channels: List[int],
        # convolution hyperparameters
        conv_kernel_sizes: List[int], conv_stride_sizes: List[int],
        conv_paddings: List[int], pool_kernel_sizes: List[int],
        pool_stride_sizes: List[int], pool_padding_sizes: List[int],
        # mlp hyperparameters
        fc_head: nn.Module|None, fc_in_size: int,
        pool: str="max", use_batchnorm: bool=False,
        conv_act_fun: nn.Module=nn.ReLU, fc_act_fun: nn.Module|None=None
    ) -> None:
        super().__init__()

        self.n_layers = len(hidden_channels)
        if hidden_channels:
            hidden_channels = [in_channels] + hidden_channels + [out_channels]
        else:
            hidden_channels = [in_channels, out_channels]
        in_channels = hidden_channels[0:-1]
        out_channels = hidden_channels[1:]

        conv_layers = []
        n_feats_prime = n_feats
        for i, (i_c, o_c, c_k, c_s, c_p, p_k, p_s, p_p) in enumerate(zip(
            in_channels, out_channels,
            conv_kernel_sizes, conv_stride_sizes, conv_paddings,
            pool_kernel_sizes, pool_stride_sizes, pool_padding_sizes,
        )):
            if i == self.n_layers:
                use_batchnorm = False
            conv_layers.append(Conv2DLayer(
                i_c, o_c, c_k, c_p, c_s, pool=None, act_fun=conv_act_fun,
                act_fun_pos="after_conv", use_batchnorm=use_batchnorm
            ))
            n_feats_prime = self.get_conv_pool_out_size(
                n_feats_prime, c_k, c_p, c_s)
            if p_k > 0:
                if pool == "max":
                    conv_layers.append(nn.MaxPool2d(p_k, p_s, padding=p_p))
                elif pool == "avg":
                    conv_layers.append(nn.AvgPool2d(p_k, p_s, padding=p_p))
                n_feats_prime = self.get_conv_pool_out_size(
                    n_feats_prime, p_k, p_p, p_s)
        
        self.conv_layers = nn.Sequential(*conv_layers)
        final_numels = hidden_channels[-1] * n_feats_prime**2
        if fc_act_fun:
            fc_map_net = nn.Sequential(
                *[nn.Linear(final_numels, fc_in_size), fc_act_fun()]
            )
        else:
            fc_map_net = nn.Sequential(
                *[nn.Linear(final_numels, fc_in_size)]
            )
        self.fc_mapper = fc_map_net
        self.fc_head = fc_head

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_mapper(x)
        if self.fc_head:
            x = self.fc_head(x)
        return x
    
    @staticmethod
    def get_conv_pool_out_size(w: int, k: int, p: int, s: int) -> int:
        """
        w: input size
        k: kernel size
        p: padding
        s: stride
        """
        w_prime = (w - k + 2*p)//s + 1
        return w_prime

class UNet(nn.Module):
    def __init__(
        self, n_feats: int, n_channels: int, hidden_channels: List[int],
        # convolution hyperparameters
        conv_kernel_sizes: List[int], conv_stride_sizes: List[int],
        conv_paddings: List[int], pool_kernel_sizes: List[int],
        pool_stride_sizes: List[int], pool_padding_sizes: List[int],
        # transpose convolution hyperparameters
        tconv_kernel_sizes: List[int], tconv_stride_sizes: List[int],
        tconv_in_paddings: List[int], tconv_out_paddings: List[int],
        pool: str="max",
        act_fun: nn.Module=nn.LogSigmoid
    ) -> None:
        assert (
            len(hidden_channels) == len(conv_kernel_sizes)
                == len(conv_stride_sizes) == len(conv_paddings)
                == len(pool_kernel_sizes) == len(pool_stride_sizes)
                == len(pool_padding_sizes)
        ), "convolution and pooling parameter lists not equal length!"
        assert (
            len(hidden_channels) == len(tconv_kernel_sizes)
            == len(tconv_stride_sizes) == len(tconv_in_paddings)
            == len(tconv_out_paddings)
        ), "transpose convolution lists not equal length!"
        super().__init__()

        self.n_layers = len(hidden_channels)
        convs = []
        pools = []
        hidden_channels = [n_channels] + hidden_channels
        in_channels = hidden_channels[0:-1]
        out_channels = hidden_channels[1:]
        n_feats_prime = n_feats
        for i_c, o_c, c_k, c_s, c_p, p_k, p_s, p_p in zip(
            in_channels, out_channels,
            conv_kernel_sizes, conv_stride_sizes, conv_paddings,
            pool_kernel_sizes, pool_stride_sizes, pool_padding_sizes,
        ):
            convs.append(Conv2DLayer(
                i_c, o_c, c_k, c_p, c_s, pool=None, act_fun=act_fun,
                act_fun_pos="after_conv"
            ))
            n_feats_prime = self.get_conv_pool_out_size(
                n_feats_prime, c_k, c_p, c_s)
            if p_k > 0:
                if pool == "max":
                    pools.append(nn.MaxPool2d(p_k, p_s, padding=p_p))
                elif pool == "avg":
                    pools.append(nn.AvgPool2d(p_k, p_s, padding=p_p))
                n_feats_prime = self.get_conv_pool_out_size(
                    n_feats_prime, p_k, p_p, p_s)
            else:
                pools.append(DummyLayer())
        
        tconvs = []
        for j, (o_c, i_c, t_k, t_s, i_p, o_p) in enumerate(zip(
            reversed(out_channels), reversed(in_channels),
            tconv_kernel_sizes, tconv_stride_sizes,
            tconv_in_paddings, tconv_out_paddings
        )):
            o_c = o_c*2 if j > 0 else o_c
            act_fun = act_fun if j < len(out_channels) - 1 else None
            tconvs.append(TransposeConv2DLayer(
                o_c, i_c, t_k, i_p, o_p, t_s, act_fun=act_fun
            ))
            n_feats_prime = self.get_tconv_out_size(
                n_feats_prime, t_k, i_p, o_p, t_s)
        
        self.convs = nn.ModuleList(convs)
        self.pools = nn.ModuleList(pools)
        self.tconvs = nn.ModuleList(tconvs)

        assert n_feats == n_feats_prime, (
            "Shape Error: Expected output shape to be the same as input "
            f"shape {n_feats} but got {n_feats_prime} instead"
        )

    def forward(self, x: torch.Tensor, cat_dim: int=1) -> torch.Tensor:
        c_xs = []
        for i, (conv, pool) in enumerate(zip(self.convs, self.pools)):
            x = conv(x)
            if i < self.n_layers - 1:
                # (B, C_in, H_in, W_in) -> (B, C_out, H_out, W_out)
                c_xs.append(x)
                x = pool(x)
        
        for j, tconv in enumerate(self.tconvs):
            if j > 0:
                # (B, 2*C_out, H_out, W_out) -> (B, C_in, H_in, W_in)
                x = torch.cat((x, c_xs[-j]), dim=cat_dim)
            x = tconv(x)
        
        return x
    
    @staticmethod
    def get_conv_pool_out_size(w: int, k: int, p: int, s: int) -> int:
        """
        w: input size
        k: kernel size
        p: padding
        s: stride
        """
        w_prime = (w - k + 2*p)//s + 1
        return w_prime

    @staticmethod
    def get_tconv_out_size(
        w_prime: int, k: int, i_p: int, o_p: int, s: int) -> int:
        """
        w_prime: output_size
        k: kernel size
        i_p: input padding
        o_p: output padding
        s: stride
        """
        w = (w_prime - 1)*s + k - 2*i_p + o_p
        return w