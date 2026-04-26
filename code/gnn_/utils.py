import torch
from torch_geometric.data import Data
from typing import Tuple


def get_triu_edge_data(batch: Data) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    row, col = batch.edge_index
    upper_mask = row < col
    lower_mask = row > col
    upper_edge_index = batch.batch[batch.edge_index[0]]
    upper_edge_attr = batch.edge_attr[upper_mask]

    return upper_edge_index, upper_edge_attr, upper_mask, lower_mask

def mirror_triu_to_tril(
    batch: Data, upper_edge_attr: torch.Tensor,
    upper_mask: torch.Tensor, lower_mask: torch.Tensor
) -> Data:
    batch = batch.clone()
    row, col = batch.edge_index

    batch.edge_attr[upper_mask] = upper_edge_attr

    upper_row = row[upper_mask]
    upper_col = col[upper_mask]

    lower_row = row[lower_mask]
    lower_col = col[lower_mask]

    upper_key = upper_row * batch.num_nodes + upper_col
    lower_key = lower_col * batch.num_nodes + lower_row

    order = torch.argsort(upper_key)
    sorted_upper_key = upper_key[order]
    sorted_upper_attr = upper_edge_attr[order]

    lower_pos = torch.searchsorted(sorted_upper_key, lower_key)
    batch.edge_attr[lower_mask] = sorted_upper_attr[lower_pos]

    return batch