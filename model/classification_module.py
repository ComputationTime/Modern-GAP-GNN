import torch
import torch.nn as nn

from .mlp import MLP


class ClassificationModule(nn.Module):
    def __init__(self, num_hops, base_dims, head_dims):
        self.base_mlps = nn.ModuleList([MLP(base_dims) for k in num_hops])
        self.head_mlp = MLP(head_dims)
        self.softmax = nn.Softmax

    def forward(self, x):
        x = [base_mlp(x[i]) for i, base_mlp in enumerate(self.base_mlps)]
        x = torch.cat(x, dim=1)
        x = self.head_mlp(x)
        x = torch.softmax(x, dim=1)
        return x
