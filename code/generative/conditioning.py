from copy import deepcopy
import torch
import torch.nn as nn
from typing import List
from nn.mlp import MLP
from nn.cnn import UNet
from nn.embedding import SinusoidalPE, ClassEmbedding


class FiLMedNetwork(nn.Module):
    def __init__(
        self, network: nn.Module|nn.Sequential,
        embedding: nn.Module|nn.Sequential, embedding_dim: int,
        feature_dim: int, act_fun: nn.Module|None=nn.ReLU
    ) -> None:
        super().__init__()

        self.emb = embedding
        self.net = network
        self.film = nn.Linear(embedding_dim, 2*feature_dim)
        if act_fun is not None:
            self.act_fun = act_fun()
        else:
            self.act_fun = None
    
    def forward(
        self, x: torch.Tensor, *conditions: torch.Tensor) -> torch.Tensor:
        c = self._get_embedding(*conditions)

        # 2 * (1/batch_size, hidden_channels)
        gamma, beta = torch.chunk(self.film(c), 2, dim=-1)
        if x.dim() == 4:
            # 2 * (1/batch_size, hidden_channels, height, width)
            gamma = gamma[:, :, None, None]
            beta = beta[:, :, None, None]
        else:
            pass
        # stabilize in case gamma is near 0
        gamma = gamma + 1.0

        x = self.net(x)
        x = x*gamma + beta
        if self.act_fun is not None:
            x = self.act_fun(x)
        
        return x

    def _get_embedding(self, c: torch.Tensor) -> torch.Tensor:
        return self.emb(c)

class FiLMHybrid(FiLMedNetwork):
    """
    Rework class a bit regarding embedding dim's and so on.
    """
    def __init__(
        self, network: nn.Module|nn.Sequential,
        continuous_embedding: nn.Module|nn.Sequential,
        class_embedding: nn.Module|nn.Sequential, embedding_dim_1: int,
        embedding_dim_2: int, feature_dim: int, act_fun: nn.Module|None=nn.ReLU
    ) -> None:
        super().__init__(
            network, continuous_embedding, embedding_dim_1, feature_dim,
            act_fun=act_fun
        )

        self.cont_emb = continuous_embedding
        self.class_emb = class_embedding
        self.mod_map = nn.Linear(
            embedding_dim_1 + embedding_dim_2, embedding_dim_1)
    
    def _get_embedding(
        self, t: torch.Tensor, y: torch.Tensor|None=None, cat_dim: int=1
    ) -> torch.Tensor:
        cont_emb, class_emb = self.cont_emb(t), self.class_emb(y)
        if class_emb.size(0) != cont_emb.size(0):
            class_emb = class_emb.expand((cont_emb.size(0), class_emb.size(1)))
        c = torch.cat([cont_emb, class_emb], dim=cat_dim)
        return self.mod_map(c)

class MLPTimeDependent(MLP):
    """
    TO-DO: Find a better solution than to overwrite the last layer again...
    """
    def __init__(
        self, n_timesteps: int, embedding_dim: int, *args,
        mlp_act_funs: List[nn.Module|None]|nn.Module|None=nn.ReLU,
        film_act_fun: nn.Module|None=nn.ReLU,
        **kwargs
    ):
        super().__init__(*args, act_funs=mlp_act_funs, **kwargs)

        new_networks = []
        for _, l_params in self.network.named_children():
            for _, s_params in l_params.named_children():
                for _, i_params in s_params.named_children():
                    if "linear" in str(type(i_params)):
                        feature_dim = i_params.out_features
                new_networks.append(FiLMedNetwork(
                    deepcopy(s_params), SinusoidalPE(
                        embedding_dim, n_timesteps), embedding_dim,
                    feature_dim, act_fun=film_act_fun
                ))
        # make sure the old network is not output-clipped by an activation
        # function
        new_networks[-1] = FiLMedNetwork(
            deepcopy(s_params), SinusoidalPE(
                embedding_dim, n_timesteps), embedding_dim,
            feature_dim, act_fun=None
        )
        
        self.network = nn.ModuleList(new_networks)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        for net in self.network:
            x = net(x, t)
        return x

class UNetTimeDependent(UNet):
    def __init__(
        self, n_timesteps: int, embedding_dim: int, *args,
        unet_act_fun: nn.Module|None=nn.ReLU,
        film_act_fun: nn.Module|None=nn.ReLU,
        **kwargs
    ) -> None:
        super().__init__(*args, act_fun=unet_act_fun, **kwargs)

        new_convs = []
        for conv in self.convs:
            for _, l_params in conv.named_children():
                for _, s_params in l_params.named_children():
                    if "conv" in str(type(s_params)):
                        new_convs.append(FiLMedNetwork(
                            deepcopy(conv), SinusoidalPE(
                                embedding_dim, n_timesteps),
                            embedding_dim, s_params.out_channels,
                            act_fun=film_act_fun
                        ))
        self.convs = nn.ModuleList(new_convs)

        new_tconvs = []
        for j, tconv in enumerate(self.tconvs):
            # make sure the old network is not output-clipped by an activation
            # function
            f_act_fun = film_act_fun if j < len(self.tconvs) - 1 else None
            for _, l_params in tconv.named_children():
                for _, s_params in l_params.named_children():
                    if "Transpose" in str(type(s_params)):
                        new_tconvs.append(FiLMedNetwork(
                            deepcopy(tconv), SinusoidalPE(
                                embedding_dim, n_timesteps),
                            embedding_dim, s_params.out_channels,
                            act_fun=f_act_fun
                        ))
        self.tconvs = nn.ModuleList(new_tconvs)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cat_dim: int=1, **kwargs
    ) -> torch.Tensor:
        c_xs = []
        for i, (conv, pool) in enumerate(zip(self.convs, self.pools)):
            x = conv(x, t)
            if i < self.n_layers - 1:
                # (B, C_in, H_in, W_in) -> (B, C_out, H_out, W_out)
                c_xs.append(x)
                x = pool(x)
        
        for j, tconv in enumerate(self.tconvs):
            if j > 0:
                # (B, 2*C_out, H_out, W_out) -> (B, C_in, H_in, W_in)
                x = torch.cat((x, c_xs[-j]), dim=cat_dim)
            x = tconv(x, t)
        
        return x

class UNetTimeClassDependent(UNet):
    def __init__(
        self, n_timesteps: int, time_emb_dim: int, class_emb_dim,
        n_classes: int, *args,
        unet_act_fun: nn.Module|None=nn.ReLU,
        film_act_fun: nn.Module|None=nn.ReLU,
        **kwargs
    ) -> None:
        super().__init__(*args, act_fun=unet_act_fun, **kwargs)

        new_convs = []
        for conv in self.convs:
            for _, l_params in conv.named_children():
                for _, s_params in l_params.named_children():
                    if "conv" in str(type(s_params)):
                        new_convs.append(FiLMHybrid(
                            deepcopy(conv), SinusoidalPE(
                                time_emb_dim, n_timesteps),
                            ClassEmbedding(class_emb_dim, n_classes),
                            time_emb_dim, class_emb_dim, s_params.out_channels,
                            act_fun=film_act_fun
                        ))
        self.convs = nn.ModuleList(new_convs)

        new_tconvs = []
        for j, tconv in enumerate(self.tconvs):
            # make sure the old network is not output-clipped by an activation
            # function
            f_act_fun = film_act_fun if j < len(self.tconvs) - 1 else None
            for _, l_params in tconv.named_children():
                for _, s_params in l_params.named_children():
                    if "Transpose" in str(type(s_params)):
                        new_tconvs.append(FiLMHybrid(
                            deepcopy(tconv), SinusoidalPE(
                                time_emb_dim, n_timesteps),
                            ClassEmbedding(class_emb_dim, n_classes),
                            time_emb_dim, class_emb_dim, s_params.out_channels,
                            act_fun=f_act_fun
                        ))
        self.tconvs = nn.ModuleList(new_tconvs)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor|None=None,
        cat_dim: int=1
    ) -> torch.Tensor:
        c_xs = []
        for i, (conv, pool) in enumerate(zip(self.convs, self.pools)):
            x = conv(x, t, y)
            if i < self.n_layers - 1:
                # (B, C_in, H_in, W_in) -> (B, C_out, H_out, W_out)
                c_xs.append(x)
                x = pool(x)
        
        for j, tconv in enumerate(self.tconvs):
            if j > 0:
                # (B, 2*C_out, H_out, W_out) -> (B, C_in, H_in, W_in)
                x = torch.cat((x, c_xs[-j]), dim=cat_dim)
            x = tconv(x, t, y)
        
        return x