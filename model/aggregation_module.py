import torch
import torch.nn as nn


def edge_index_to_adjacency_matrix(edge_index, n):
    edge_attr = torch.ones(edge_index.size(1))
    A = torch.sparse_coo_tensor(edge_index, edge_attr, size=(n, n)).coalesce()
    return A


class PrivateMultihopAggregation(nn.Module):
    def __init__(self, num_hops, noise_scale):
        super().__init__()
        self.num_hops = num_hops
        self.noise_scale = noise_scale

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        A = edge_index_to_adjacency_matrix(edge_index, x.size(0))
        out = [nn.functional.normalize(x, p=2, dim=1)]
        for k in range(self.num_hops):
            noise = torch.randn(*x.size()) * self.noise_scale
            x_k = torch.mm(A, out[k])
            x_k = torch.add(x_k, noise)
            x_k = nn.functional.normalize(x_k, p=2, dim=1)
            out.append(x_k)
        return torch.stack(out)
