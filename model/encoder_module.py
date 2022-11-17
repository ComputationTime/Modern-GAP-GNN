import torch.nn as nn
from torchvision.ops import MLP


def EncoderModule(dimensions):
    return MLP(
        in_channels=dimensions[0],
        hidden_channels=dimensions[1:],
        norm_layer=None,
        activation_layer=nn.ReLU,
        inplace=True,
        bias=True,
        dropout=0
    )
