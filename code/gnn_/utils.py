import torch
from torch_geometric.data import Data
from typing import Tuple


def extract_triu_node_data(
    edge_index: torch.Tensor, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    row, col = edge_index
    upper_mask = row < col
    upper_row, upper_col = row[upper_mask], col[upper_mask]
    x_src, x_dst = x[upper_row], x[upper_col]
    return x_src, x_dst

def extract_triu_edge_data(
    edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
    row, col = edge_index
    upper_edge_attr = edge_attr[row < col]
    return upper_edge_attr

def mirror_triu_to_tril_again(
    edge_index: torch.Tensor, edge_attr: torch.Tensor,
    upper_edge_attr: torch.Tensor, n_nodes: int
) -> torch.Tensor:
    edge_attr = edge_attr.clone()

    row, col = edge_index
    upper_mask = row < col
    lower_mask = row > col

    upper_row = row[upper_mask]
    upper_col = col[upper_mask]
    lower_row = row[lower_mask]
    lower_col = col[lower_mask]

    upper_key = upper_row * n_nodes + upper_col
    lower_key = lower_col * n_nodes + lower_row

    order = torch.argsort(upper_key)
    sorted_upper_key = upper_key[order]
    sorted_upper_attr = upper_edge_attr[order]

    lower_pos = torch.searchsorted(sorted_upper_key, lower_key)

    edge_attr[upper_mask] = upper_edge_attr
    edge_attr[lower_mask] = sorted_upper_attr[lower_pos]

    return edge_attr