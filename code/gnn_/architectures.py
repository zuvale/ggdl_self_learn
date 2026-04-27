import torch
import torch.nn as nn
import torch_geometric.nn as g_nn
from torch_geometric.data import Data
from typing import Tuple

from nn_.mlp import MLP
from nn_.embedding import ClassEmbedding, ContinuousSinusoidalPE
from generative.conditioning import FiLMHybrid
from gnn_.utils import (
    extract_triu_node_data, extract_triu_edge_data, mirror_triu_to_tril_again)


class GNNDenoiser(nn.Module):
    def __init__(
        self, node_feat: int, edge_feat: int, node_hidden: int,
        edge_hidden: int, time_emb_dim: int,
        class_emb_dim: int, n_classes: int, n_layers: int=1,
        updater_layers: int=2, updater_size: int=4, input_type: str="tokens",
        message_passing: str="gineconv", act_fun: nn.Module=nn.GELU,
        loader=None
    ) -> None:
        super().__init__()

        self.n_layers = n_layers

        self.input_type = input_type
        if self.input_type == "tokens":
            self.node_emb = nn.Embedding(node_feat, node_hidden)
            self.edge_emb = nn.Embedding(edge_feat, edge_hidden)
        elif self.input_type == "probs":
            self.node_proj_in = nn.Linear(node_feat, node_hidden)
            self.edge_proj_in = nn.Linear(edge_feat, edge_hidden)

        self.mp_convolutions = nn.ModuleList()
        self.edge_mlps = nn.ModuleList()

        self.node_filmers = nn.ModuleList()
        self.edge_filmers = nn.ModuleList()

        self.node_proj_out = nn.Linear(node_hidden, node_feat)
        self.edge_proj_out = nn.Linear(edge_hidden, edge_feat)

        self.node_norms = nn.ModuleList()
        self.edge_norms = nn.ModuleList()

        for k in range(self.n_layers):
            
            self.mp_convolutions.append(self.conv_create_wrapper(
                message_passing, node_hidden, edge_hidden, updater_size,
                updater_layers, act_fun, loader
            ))
            self.edge_mlps.append(
               MLP(
                    2 * node_hidden + edge_hidden, edge_hidden,
                    [edge_hidden*updater_size]*updater_layers,
                    act_funs=[act_fun]*(updater_layers + 1)
                ) 
            )

            self.node_filmers.append(
                FiLMHybrid(
                    MLP(
                        node_hidden, node_hidden,
                        [node_hidden*updater_size]*updater_layers,
                        act_funs=[act_fun]*(updater_layers + 1)
                    ),
                    ContinuousSinusoidalPE(time_emb_dim),
                    ClassEmbedding(class_emb_dim, n_classes), time_emb_dim,
                    class_emb_dim, node_hidden, act_fun=act_fun
                )
            )
            self.edge_filmers.append(
                FiLMHybrid(
                    MLP(
                        edge_hidden, edge_hidden,
                        [edge_hidden*updater_size]*updater_layers,
                        act_funs=[act_fun]*(updater_layers + 1)
                    ),
                    ContinuousSinusoidalPE(time_emb_dim),
                    ClassEmbedding(class_emb_dim, n_classes), time_emb_dim,
                    class_emb_dim, edge_hidden, act_fun=act_fun
                )
            )

            self.node_norms.append(nn.LayerNorm(node_hidden))
            self.edge_norms.append(nn.LayerNorm(edge_hidden))
    
    def forward(
        self, batch: Data, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        e_idx, y = batch.edge_index, batch.y
        row, col = e_idx
        upper_graph = batch.batch[row[row < col]]

        # 1. Project Data
        if self.input_type == "tokens":
            # 1.A Embed Tokens to Hidden Representation
            hid_node = self.node_emb(self._as_token_ids(batch.x))
            hid_edge = self.edge_emb(self._as_token_ids(batch.edge_attr))
        elif self.input_type == "probs":
            # 1.B Project Probabilities to Hidden Representation
            hid_node = self.node_proj_in(batch.x.float())
            hid_edge = self.edge_proj_in(batch.edge_attr.float())

        # 2. Message-Passing/Updating and Modulation
        for k in range(self.n_layers):
            # 2.1 Nodes
            # 2.1.1 Pass Messages Around
            node_update = self.mp_convolutions[k](
                hid_node, e_idx, edge_attr=hid_edge)
            # 2.1.2 Cast Time and Class to Node Level and Modulate
            t_node, y_node = t[batch.batch], y[batch.batch]
            node_update = self.node_filmers[k](node_update, t_node, y_node)
            # 2.1.3 Residual Connection & Layer Normalization
            hid_node = self.node_norms[k](hid_node + node_update)

            # 2.2 Edges
            # 2.2.1 Extract Upper-Triangular Part
            h_n_src, h_n_tgt = extract_triu_node_data(e_idx, hid_node)
            h_triu_e = extract_triu_edge_data(e_idx, hid_edge)
            # 2.2.2 Update Hidden Edge Features
            triu_e_update = self.edge_mlps[k](torch.cat([
                h_n_src + h_n_tgt, torch.abs(h_n_src - h_n_tgt), h_triu_e
            ], dim=-1))
            # 2.2.3 Cast Time and Class to Edge Level and Modulate
            t_u_e, y_u_e = t[upper_graph], y[upper_graph]
            triu_e_update = self.edge_filmers[k](triu_e_update, t_u_e, y_u_e)
            # 2.2.4 Residual Connection & Layer Normalization
            h_triu_e = self.edge_norms[k](h_triu_e + triu_e_update)
            # 2.2.5 Mirror Final Features Back to "Full Adjacency"
            hid_edge = mirror_triu_to_tril_again(
                e_idx, hid_edge, h_triu_e, batch.num_nodes)
        
        # 3. Project Hidden Embedding To Logits
        node_logits = self.node_proj_out(hid_node)
        edge_logits = self.edge_proj_out(hid_edge)

        return node_logits, edge_logits
    
    def _as_token_ids(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.argmax(dim=-1)
        return z.long()
    
    @staticmethod
    def conv_create_wrapper(
        conv_name: str, node_hidden: int, edge_hidden: int, updater_size: int,
        updater_layers: int, act_fun: nn.Module, loader
    ) -> g_nn.MessagePassing:
        if conv_name == "gineconv":
            mp_nn = MLP(
                node_hidden, node_hidden,
                [node_hidden*updater_size]*updater_layers,
                act_funs=[act_fun]*(updater_layers + 1)
            )
            mp_nn.in_channels = node_hidden

            return g_nn.GINEConv(mp_nn, edge_dim=edge_hidden)

        if conv_name == "pna_conv":
            # needs some more tuning and more arguments from __init__
            deg = g_nn.PNAConv.get_degree_histogram(loader)
            return g_nn.PNAConv(
                node_hidden, node_hidden, ["mean", "min", "max", "std"],
                ["identity", "amplification", "attenuation"], deg,
                edge_dim=edge_hidden, act="gelu"
            )