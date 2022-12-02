import torch
import torch.nn as nn


class PrivateMultihopAggregation(nn.Module):
    def __init__(self, num_hops, noise_scale):
        super().__init__()
        self.num_hops = num_hops
        self.noise_scale = noise_scale

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        out = [nn.functional.normalize(x, p=2, dim=1)]
        for k in range(self.num_hops):
            noise = torch.randn(*x.size()) * self.noise_scale
            x_k = torch.mm(A, out[k])
            x_k = torch.add(x_k, noise)
            x_k = nn.functional.normalize(x_k, p=2, dim=1)
            out.append(x_k)
        return torch.stack(out)
