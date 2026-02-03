from nn import mlp, cnn
import torch
import torch.nn as nn


class MNISTClassifier(cnn.CNN2DToFC):
    def forward(
        self, x: torch.Tensor, return_final_feats: bool=False
    ) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x_feats = self.fc_mapper(x)
        x_logits = self.fc_head(x_feats)

        if not return_final_feats:
            return x_logits
        else:
            return x_feats

def create_mnist_classifier() -> nn.Module:
    return MNISTClassifier(
        28, 1, 64, [32, 32, 64],
        [3, 3, 3, 3], [1, 1, 1, 1], [1, 1, 1, 1],
        [0, 2, 0, 2], [0, 2, 0, 2], [0, 0, 0, 0],
        mlp.MLP(256, 10, act_funs=None), 256
    )
