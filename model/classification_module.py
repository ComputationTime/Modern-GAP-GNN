import torch
import torch.nn as nn

from .mlp import MLP


class MLPClassifier(nn.Module):
    def __init__(self, num_hops, base_dims, head_dims):
        super().__init__()
        self.base_mlps = nn.ModuleList([MLP(base_dims) for _ in range(num_hops+1)])
        self.head_mlp = MLP(head_dims)
        self.softmax = nn.Softmax

    def forward(self, x):
        x = [base_mlp(x[i]) for i, base_mlp in enumerate(self.base_mlps)]
        x = torch.cat(x, dim=1)
        x = self.head_mlp(x)
        x = torch.softmax(x, dim=1)
        return x
