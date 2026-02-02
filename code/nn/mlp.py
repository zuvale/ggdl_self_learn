import torch
import torch.nn as nn


class FCLayer(nn.Module):
    """
    WIP: add dropout
    """
    def __init__(
            self, in_feats: int, out_feats: int, use_batchnorm: bool=True,
            act_fun: nn.modules.activation=nn.ReLU
        ) -> None:
        super().__init__()

        modules = [nn.Linear(in_feats, out_feats)]
        if use_batchnorm:
            modules.append(nn.BatchNorm1d(out_feats))
        if act_fun is not None:
            modules.append(act_fun())
        
        self.fc_layer = nn.Sequential(*modules)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_layer(x)

class MLP(nn.Module):
    """
    WIP: add dropout
    """
    def __init__(
            self, n_feats: int, out_size: int, 
            hidden_sizes: None|int|list[int]=None, use_batchnorm: bool=False,
            act_funs: list[nn.Module]|nn.Module=nn.ReLU, flatten=False
        ) -> None:
        super().__init__()

        if hidden_sizes:
            if isinstance(hidden_sizes, int):
                hidden_sizes = [hidden_sizes]
            # parameterize the number of hidden layers and nodes
            layer_sizes = [n_feats] + hidden_sizes + [out_size]
            n_layers = len(layer_sizes) - 1
            in_layer_sizes = layer_sizes[0:n_layers]
            out_layer_sizes = layer_sizes[1:n_layers + 1]

            # parameterize activation functions
            if not isinstance(act_funs, list):
                act_funs = [act_funs]*n_layers

            fc_block_list = [
                FCLayer(
                    n_in, n_out, use_batchnorm=use_batchnorm, act_fun=a_fun)
                for n_in, n_out, a_fun
                in zip(in_layer_sizes, out_layer_sizes, act_funs)
            ]
        else:
            fc_block_list = [FCLayer(
                n_feats, out_size, use_batchnorm=use_batchnorm,
                act_fun=act_funs
            )]
        
        if isinstance(flatten, bool) and flatten == True:
            fc_block_list.insert(0, nn.Flatten())
        elif not isinstance(flatten, bool):
            fc_block_list.append(nn.Unflatten(-1, flatten))
        
        self.network = nn.Sequential(*fc_block_list)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)