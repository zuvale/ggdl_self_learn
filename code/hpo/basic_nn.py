from mealpy import Problem, FloatVar, IntegerVar, StringVar, MixedSetVar
import torch.nn as nn
from typing import Iterable
from nn.cnn import UNet
from generative.conditioning import UNetTimeDependent, UNetTimeClassDependent
from .utils import powers_of_2

POWERS_OF_2 = powers_of_2(11)
UNET_BOUNDS = [
    ## ---- Encoder Parameters ---
    IntegerVar(lb=1, ub=5, name="n_hidden_layers"),
    # convolution 
    MixedSetVar(valid_sets=POWERS_OF_2, name="hid_chan_size_1"),
    MixedSetVar(valid_sets=POWERS_OF_2, name="hid_chan_size_2"),
    MixedSetVar(valid_sets=POWERS_OF_2, name="hid_chan_size_3"),
    MixedSetVar(valid_sets=POWERS_OF_2, name="hid_chan_size_4"),
    MixedSetVar(valid_sets=POWERS_OF_2, name="hid_chan_size_5"),
    MixedSetVar(valid_sets=POWERS_OF_2, name="hid_chan_size_6"),
    MixedSetVar(valid_sets=POWERS_OF_2, name="hid_chan_size_7"),
    MixedSetVar(valid_sets=(3, 5, 7), name="conv_kernel_size"),
    # pooling
    MixedSetVar(valid_sets=(2, 3, 4), name="pool_kernel_size"),
    IntegerVar(lb=2, ub=2, name="pool_stride_size"),
    IntegerVar(lb=0, ub=1, name="pool_pad_size_1"),
    IntegerVar(lb=0, ub=1, name="pool_pad_size_2"),
    IntegerVar(lb=0, ub=1, name="pool_pad_size_3"),
    IntegerVar(lb=0, ub=1, name="pool_pad_size_4"),
    IntegerVar(lb=0, ub=1, name="pool_pad_size_5"),
    IntegerVar(lb=0, ub=1, name="pool_pad_size_6"),
    IntegerVar(lb=0, ub=1, name="pool_pad_size_7"),
    ## --- Decoder Parameters ---
    # transpose convolution
    MixedSetVar(valid_sets=(3, 5, 7), name="tconv_kernel_size"),
    ## --- Other UNet Parameters
    StringVar(
        valid_sets=(
            "relu", "relu6", "prelu", "rrelu", "selu", "silu", "gelu"),
        name="u_act_fun"
    )
]
OPTIM_BOUNDS = [
    FloatVar(lb=0.001, ub=0.01, name="learning_rate"),
    StringVar(
            valid_sets=("adam", "adamw", "nadam", "radam"), name="optim_fun"),
    FloatVar(lb=0.90, ub=1.00, name="gamma")
]


def create_unet(
    input_channels: int, n_feats: int, hidden_channel_sizes: Iterable[int],
    num_hidden_layers: int, conv_kernel_size: int, pool_kernel_size: int,
    pool_stride_size: int, pool_padding_sizes: Iterable[int],
    transpose_conv_kernel_size: int,
    *other_args,
    activation_function: nn.Module|None=None, unet_type: str="vanilla",
    **other_kwargs
) -> UNet|UNetTimeDependent|UNetTimeClassDependent:
    hidden_channels = [input_channels] + hidden_channel_sizes
    in_channels = hidden_channels[0:-1]
    out_channels = hidden_channels[1:]

    conv_kernel_sizes = [conv_kernel_size] * num_hidden_layers
    conv_stride_sizes = [1] * num_hidden_layers
    conv_padding_sizes = [conv_kernel_size//2] * num_hidden_layers

    pool_kernel_sizes = [pool_kernel_size] * num_hidden_layers
    pool_kernel_sizes[-1] = 0
    pool_stride_sizes = [pool_stride_size] * num_hidden_layers

    tconv_kernel_sizes = [transpose_conv_kernel_size] * num_hidden_layers
    tconv_in_pad_sizes = [transpose_conv_kernel_size//2] * num_hidden_layers

    n_feats_prime = n_feats
    encoder_sizes = []
    for i_c, o_c, c_k, c_s, c_p, p_k, p_s, p_p in zip(
        in_channels, out_channels,
        conv_kernel_sizes, conv_stride_sizes, conv_padding_sizes,
        pool_kernel_sizes, pool_stride_sizes, pool_padding_sizes,
    ):
        n_feats_prime = UNet.get_conv_pool_out_size(
            n_feats_prime, c_k, c_p, c_s)
        
        if p_k > 0:
            n_feats_prime = UNet.get_conv_pool_out_size(
                n_feats_prime, p_k, p_p, p_s)
        
        encoder_sizes.append(n_feats_prime)

    target_sizes = sorted(list(set([n_feats] + encoder_sizes)))[1:]
    if len(target_sizes) < num_hidden_layers:
        target_sizes.append(n_feats)
    
    tconv_stride_sizes = []
    tconv_out_pad_sizes = []
    for j, (o_c, i_c, t_k, i_p, tgt) in enumerate(zip(
        reversed(out_channels), reversed(in_channels),
        tconv_kernel_sizes, tconv_in_pad_sizes, target_sizes
    )):
        t_s = 2
        out_pad = tgt - ((n_feats_prime - 1) * t_s - 2*i_p + t_k)
        if out_pad < 0:
            t_s = 1
            out_pad = tgt - ((n_feats_prime - 1) * t_s - 2*i_p + t_k)

        if out_pad > 1:
            t_s = 2
            t_k = max(t_k - 2, 3)
            i_p = min(i_p, 0)
            #t_conv_in_pad_sizes[j] = i_p
            out_pad = tgt - ((n_feats_prime - 1) * t_s - 2*i_p + t_k)
        
        if out_pad < 0:
            t_s = 2
            i_p = max(i_p, 2)
            out_pad = tgt - ((n_feats_prime - 1) * t_s - 2*i_p + t_k)
        
        tconv_kernel_sizes[j] = t_k
        tconv_in_pad_sizes[j] = i_p
        tconv_stride_sizes.append(t_s)
        tconv_out_pad_sizes.append(out_pad)
        n_feats_prime = tgt
    
    if unet_type == "vanilla":
        return UNet(
            n_feats, input_channels, hidden_channel_sizes[:num_hidden_layers],
            conv_kernel_sizes, conv_stride_sizes, conv_padding_sizes,
            pool_kernel_sizes, pool_stride_sizes,
            pool_padding_sizes[:num_hidden_layers],
            tconv_kernel_sizes, tconv_stride_sizes, tconv_in_pad_sizes,
            tconv_out_pad_sizes, act_fun=activation_function
        )
    elif unet_type == "time_filmed":
        return UNetTimeDependent(
            *other_args,
            n_feats, input_channels, hidden_channel_sizes[:num_hidden_layers],
            conv_kernel_sizes, conv_stride_sizes, conv_padding_sizes,
            pool_kernel_sizes, pool_stride_sizes,
            pool_padding_sizes[:num_hidden_layers],
            tconv_kernel_sizes, tconv_stride_sizes, tconv_in_pad_sizes,
            tconv_out_pad_sizes, unet_act_fun=activation_function,
            **other_kwargs
        )
    elif unet_type == "time_class_filmed":
        return UNetTimeClassDependent(
            *other_args,
            n_feats, input_channels, hidden_channel_sizes[:num_hidden_layers],
            conv_kernel_sizes, conv_stride_sizes, conv_padding_sizes,
            pool_kernel_sizes, pool_stride_sizes,
            pool_padding_sizes[:num_hidden_layers],
            tconv_kernel_sizes, tconv_stride_sizes, tconv_in_pad_sizes,
            tconv_out_pad_sizes, unet_act_fun=activation_function,
            **other_kwargs
        )