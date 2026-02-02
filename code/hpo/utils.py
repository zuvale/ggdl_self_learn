import numpy as np
import torch.nn as nn
import torch.optim as optim


def get_act_fun(name):
    match name:
        case "relu":
            return nn.ReLU
        case "relu6":
            return nn.ReLU6
        case "prelu":
            return nn.PReLU
        case "rrelu":
            return nn.RReLU
        case "selu":
            return nn.SELU
        case "silu":
            return nn.SiLU
        case "gelu":
            return nn.GELU

def get_optim(name):
    match name:
        case "adam":
            return optim.Adam
        case "adamw":
            return optim.AdamW
        case "nadam":
            return optim.NAdam
        case "radam":
            return optim.RAdam

powers_of_2 = lambda i: (2 ** np.arange(i))[1:]