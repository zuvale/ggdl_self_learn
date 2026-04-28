## TO-DO:
# - make refactor the FILMedCNN class with the FiLMedNetwork class to reduce
#   boilerplate
from copy import deepcopy
import torch
import torch.nn as nn
from typing import List
from nn_.mlp import MLP
from nn_.cnn import CNN2DToFC, TCNN2DFromFC, UNet
from nn_.embedding import (
    DiscreteSinusoidalPE, ContinuousSinusoidalPE, ClassEmbedding)


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

class FiLMedMultipleNetwork(FiLMedNetwork):
    def __init__(
        self, network: nn.Module|nn.Sequential, embedding_dim: int,
        feature_dim: int, *embeddings: nn.Module|nn.Sequential,
        act_fun: nn.Module|None=nn.ReLU
    ) -> None:
        super().__init__(
            network, embeddings[0], embedding_dim, feature_dim,
            act_fun=act_fun
        )
        self.embs = nn.ModuleList(embeddings)
        self.mod_map = nn.Linear(embedding_dim*2, embedding_dim)
    
    def _get_embedding(
        self, *conditions: torch.Tensor, cat_dim: int=1) -> torch.Tensor:
        embs = []
        for i, c in enumerate(conditions):
            embs.append(self.embs[i](c))
        
        return self.mod_map(torch.cat(embs, dim=cat_dim))
    
class FILMedCNN(nn.Module):
    def __init__(
        self, conv_net: nn.Module|nn.Sequential, n_classes: int,
        condition_dim: int, input_dim: int, hidden_dim: int,
        n_convs: int=4, act_fun=nn.ReLU
    ) -> None:
        super().__init__()

        self.proj_in = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)

        nets = []
        for _ in range(n_convs):
            nets.append(deepcopy(conv_net))
            if act_fun is not None:
                nets.append(act_fun())
        self.conv_net = nn.Sequential(*nets)
        self.class_embed = nn.Parameter(
            torch.randn(n_classes, condition_dim)*1e-2)
        self.film = nn.Linear(condition_dim, 2*hidden_dim)
        if act_fun is not None:
            self.act_fun = act_fun()
        else:
            self.act_fun = None
        
        self.proj_out = nn.Conv2d(hidden_dim, 2*input_dim, kernel_size=1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        
    def forward(
        self, x: torch.Tensor, y: torch.Tensor|None=None) -> torch.Tensor:
        # (batch_size, no_of_channels, height, width)
        # -> (batch_size, hidden_channels, height, width)
        x = self.proj_in(x)

        # (batch_size, cond_dim)
        if y is not None:
            # if some classes are missing (e.g. for unconditional training
            # with dropout)
            if (y < 0).any():
                m = (y < 0).float().view(-1, 1)
                class_embed = (
                    (1 - m) * self.class_embed[y]
                        + m * (
                            self.class_embed.mean(dim=0)
                                .unsqueeze(0)
                                .expand(x.size(0), -1)
                        )
                )
            else:
                class_embed = self.class_embed[y]
            c = class_embed
        else:
            c = (
                self.class_embed.mean(dim=0)
                    .unsqueeze(0)
                    .expand(x.size(0), -1)
            )
        
        # 2 * (batch_size, hidden_channels)
        gamma, beta = torch.chunk(self.film(c), 2, dim=-1)
        if x.dim() == 4:
            # 2 * (batch_size, hidden_channels, height, width)
            gamma = gamma[:, :, None, None]
            beta = beta[:, :, None, None]
        else:
            pass
        # stabilize in case gamma is near 0
        gamma = gamma + 1.0

        # (batch_size, hidden_channels, height, width)
        x = self.conv_net(x)
        x = x*gamma + beta
        if self.act_fun is not None:
            x = self.act_fun(x)
        # (batch_size, 2*no_of_channels, height, width)
        x = self.proj_out(x)
        
        # 2 * (batch_size, hidden_channels, height, width)
        return torch.chunk(x, 2, dim=1)

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

class FiLMHybrid2(FiLMedNetwork):
    """
    Rework class a bit regarding embedding dim's and so on.
    """
    def __init__(
        self, network: nn.Module|nn.Sequential, feature_dim: int,
        class_emb_dim: int, cont_emb_dim: int,
        class_embedding: nn.Module|nn.Sequential,
        *continuous_embeddings: nn.Module|nn.Sequential,
         act_fun: nn.Module|None=nn.ReLU
    ) -> None:
        super().__init__(
            network, class_embedding, class_emb_dim, feature_dim,
            act_fun=act_fun
        )

        self.class_emb = class_embedding
        self.cont_embs = nn.ModuleList(continuous_embeddings)
        input_size = sum(
            [class_emb_dim] + [cont_emb_dim] * len(continuous_embeddings))
        self.mod_map = nn.Linear(input_size, class_emb_dim)
    
    def _get_embedding(
        self, *cont_vars: torch.Tensor, y: torch.Tensor|None=None,
        cat_dim: int=1
    ) -> torch.Tensor:
        cont_embs = []
        for i, cv in enumerate(cont_vars):
            cont_embs.append(self.cont_embs[i](cv))
        class_emb = self.class_emb(y)
        if class_emb.size(0) != cont_embs[0].size(0):
            class_emb = class_emb.expand(
                (cont_embs[0].size(0), class_emb.size(1)))
        c = torch.cat(*cont_embs + [class_emb], dim=cat_dim)
        return self.mod_map(c)

class MLPTimeDependent(MLP):
    """
    TO-DO: Find a better solution than to overwrite the last layer again...
    """
    def __init__(
        self, n_timesteps: int, embedding_dim: int, *args,
        continuous: bool=False, time_scale: float=1.0, n_vars: int=1,
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
                new_networks.append(self._film_network(
                    s_params, continuous, feature_dim, embedding_dim,
                    n_timesteps, time_scale, film_act_fun, n_vars
                ))
        # make sure the old network is not output-clipped by an activation
        # function
        new_networks[-1] = self._film_network(
            s_params, continuous, feature_dim, embedding_dim,
            n_timesteps, time_scale, None, n_vars
        )
        
        self.network = nn.ModuleList(new_networks)
    
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        for net in self.network:
            x = net(x, t, *args)
        return x
    
    def _film_network(
        self, s_params, c_flag, feat_dim, emb_dim, n_t, t_scale, act_fun,
        n_vars
    ) -> nn.Module:
        if n_vars == 1:
            return FiLMedNetwork(
                deepcopy(s_params), _make_time_embedding(
                    c_flag, emb_dim, n_t, t_scale),
                emb_dim, feat_dim, act_fun=act_fun
            )
        else:
            return FiLMedMultipleNetwork(
                deepcopy(s_params), emb_dim, feat_dim, _make_time_embedding(
                    c_flag, emb_dim, n_t, t_scale), _make_time_embedding(
                    c_flag, emb_dim, n_t, t_scale),
                act_fun=act_fun
            )

class FiLMedCNN2DToFC(CNN2DToFC):
    def __init__(
        self, embedding_dim: int, n_classes: int, *args,
        film_act_fun: nn.Module|None=nn.ReLU, **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        new_convs = []
        for conv in self.conv_layers:
            for _, l_params in conv.named_children():
                for _, s_params in l_params.named_children():
                    if "conv" in str(type(s_params)):
                        new_convs.append(FiLMedNetwork(
                            deepcopy(conv), ClassEmbedding(
                                embedding_dim, n_classes),
                            embedding_dim, s_params.out_channels,
                            act_fun=film_act_fun
                        ))
        self.conv_layers = nn.ModuleList(new_convs)

        new_mapper = []
        for _,l in self.fc_mapper.named_children():
            if isinstance(l, nn.modules.linear.Linear):
                self.filmed_mapper = FiLMedNetwork(
                    deepcopy(l), ClassEmbedding(embedding_dim, n_classes),
                    embedding_dim, l.out_features, act_fun=film_act_fun
                )
            else:
                new_mapper.append(l)
    
        self.fc_mapper = nn.Sequential(*new_mapper)
    
    def forward(
        self, x: torch.Tensor, y: torch.Tensor|None=None) -> torch.Tensor:
        for conv in self.conv_layers:
            x = conv(x, y)
        x = torch.flatten(x, start_dim=1)
        x = self.filmed_mapper(x, y)
        x = self.fc_mapper(x)
        if self.fc_head:
            x = self.fc_head(x)
        return x

class FiLMedTCNN2DFromFC(TCNN2DFromFC):
    def __init__(
        self, embedding_dim: int, n_classes: int, *args,
        film_act_fun: nn.Module|None=nn.ReLU, **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        new_mapper = []
        for _,l in self.fc_mapper.named_children():
            if isinstance(l, nn.modules.linear.Linear):
                self.filmed_mapper = FiLMedNetwork(
                    deepcopy(l), ClassEmbedding(embedding_dim, n_classes),
                    embedding_dim, l.out_features, act_fun=film_act_fun
                )
            else:
                new_mapper.append(l)
    
        self.fc_mapper = nn.Sequential(*new_mapper)

        new_tconvs = []
        for j, tconv in enumerate(self.tconv_layers):
            # make sure the old network is not output-clipped by an activation
            # function
            f_act_fun = film_act_fun if j < len(self.tconv_layers) - 1 else None
            for _, l_params in tconv.named_children():
                for _, s_params in l_params.named_children():
                    if "Transpose" in str(type(s_params)):
                        new_tconvs.append(FiLMedNetwork(
                            deepcopy(tconv), ClassEmbedding(
                                embedding_dim, n_classes),
                            embedding_dim, s_params.out_channels,
                            act_fun=f_act_fun
                        ))
        self.tconv_layers = nn.ModuleList(new_tconvs)
    
    def forward(
        self, x: torch.Tensor, y: torch.Tensor|None=None) -> torch.Tensor:
        x = self.filmed_mapper(x, y)
        x = self.fc_mapper(x)
        for tconv in self.tconv_layers:
            x = tconv(x, y)
        return x

class UNetTimeDependent(UNet):
    def __init__(
        self, n_timesteps: int, embedding_dim: int, *args,
        continuous: bool=False, time_scale: float=1.0,
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
                            deepcopy(conv), _make_time_embedding(
                                continuous, embedding_dim, n_timesteps,
                                time_scale
                            ), embedding_dim, s_params.out_channels,
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
                            deepcopy(tconv), _make_time_embedding(
                                continuous, embedding_dim, n_timesteps,
                                time_scale
                            ), embedding_dim, s_params.out_channels,
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
        n_classes: int, *args, continuous: bool=False, time_scale: float=1.0,
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
                            deepcopy(conv), _make_time_embedding(
                                continuous, time_emb_dim, n_timesteps,
                                time_scale
                            ), ClassEmbedding(class_emb_dim, n_classes),
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
                            deepcopy(tconv), _make_time_embedding(
                                continuous, time_emb_dim, n_timesteps,
                                time_scale
                            ), ClassEmbedding(class_emb_dim, n_classes),
                            time_emb_dim, class_emb_dim, s_params.out_channels,
                            act_fun=f_act_fun
                        ))
        self.tconvs = nn.ModuleList(new_tconvs)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, *args,
        y: torch.Tensor|None=None, cat_dim: int=1, **kwargs
    ) -> torch.Tensor:
        c_xs = []
        for i, (conv, pool) in enumerate(zip(self.convs, self.pools)):
            x = conv(x, t, y, **args)
            if i < self.n_layers - 1:
                # (B, C_in, H_in, W_in) -> (B, C_out, H_out, W_out)
                c_xs.append(x)
                x = pool(x)
        
        for j, tconv in enumerate(self.tconvs):
            if j > 0:
                # (B, 2*C_out, H_out, W_out) -> (B, C_in, H_in, W_in)
                x = torch.cat((x, c_xs[-j]), dim=cat_dim)
            x = tconv(x, t, y, **args)
        
        return x

def _make_time_embedding(
    cont_flag: bool, emb_dim: int, tsteps: int, tscale: float) -> nn.Module:
    if cont_flag:
        return ContinuousSinusoidalPE(emb_dim, time_scale=tscale)
    else:
        return DiscreteSinusoidalPE(emb_dim, tsteps)