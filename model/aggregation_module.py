import torch
import torch.nn as nn

from .mlp import MLP


def edge_index_to_adjacency_matrix(edge_index, n, device):
    edge_attr = torch.ones(edge_index.size(1), device=device)
    A = torch.sparse_coo_tensor(edge_index, edge_attr, size=(n, n), device=device).coalesce()
    return A


class PMA(nn.Module):
    def __init__(self, num_hops, noise_scale):
        super().__init__()
        self.num_hops = num_hops
        self.noise_scale = noise_scale

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        A = edge_index_to_adjacency_matrix(edge_index, x.size(0), x.device)
        out = [nn.functional.normalize(x, p=2, dim=1)]
        for k in range(self.num_hops):
            noise = torch.randn(*x.size(), device=x.device) * self.noise_scale
            x_k = torch.mm(A, out[k])
            x_k = torch.add(x_k, noise)
            x_k = nn.functional.normalize(x_k, p=2, dim=1)
            out.append(x_k)
        return torch.stack(out)


class PMAT(nn.Module):
    def __init__(self, num_hops, encoding_dimensions, sigma):
        super().__init__()
        self.num_hops = num_hops
        self.sigma = sigma
        self.sigmoid = nn.Sigmoid()
        self.attentions = nn.ModuleList()
        for i in range(num_hops):
            # self.transforms.append(nn.Linear(*transform_dimensions)) # Only 1 layer transformation
            # self.attentions.append(MLP([2*transform_dimensions[-1], 1])) # Attention mechanism takes 2 encodings and outputs 1 weight
            # TODO: Figure out if we want a transformer?
            self.attentions.append(MLP([2*encoding_dimensions, 1]))

    def forward(self, x, edge_index):
        out = [torch.nn.functional.normalize(x, dim=1)]
        for k in range(self.num_hops):
            # Do we need to do a transform? I reckon we can use raw encoding and aggregate according to attention (and then the Classification module)
            # can handle how the aggregations get transformed
            # h = self.transforms[k](out[-1])
            h = out[-1]
            e_values = self.attentions[k](
                h[edge_index.T].reshape(edge_index.size(dim=1), 2*h.size(dim=1))
            )  # DPSGD to guarantee DP attention training
            # we have to use Sigmoid because if we use Softmax, removing an edge will change the weight of all other edges in the neighbourhood
            alpha_values = self.sigmoid(e_values)
            alpha = torch.sparse_coo_tensor(
                edge_index,
                alpha_values.reshape(
                    edge_index.size(dim=1)),
                (x.size(dim=0), x.size(dim=0)),
                dtype=torch.float
            ).transpose(0, 1)

            aggr = torch.sparse.mm(alpha, h)
            # Might need to not use "transforms" and instead do raw aggregations like the original PMA
            # aggr = torch.mm(self.A, out[-1])
            # noised = aggr # TODO: add noise # Gaussian mechanism to guarantee DP for neighbourhood aggregation
            noised = aggr + torch.normal(torch.zeros(aggr.size()), std=self.sigma).to(x.device)
            normalized = torch.nn.functional.normalize(noised, dim=1)
            out.append(normalized)
        return torch.stack(out)


class Similarity(nn.Module):
    def __init__(self, function_name: str):
        super().__init__()
        if function_name == "inner product":
            self.fun = lambda x: torch.sum(x[0, :]*x[1, :], dim=1)
        else:
            raise NotImplementedError("Not a valid similarity function")

    def forward(self, x):
        return self.fun(x)


class PMWA(nn.Module):
    # def __init__(self, num_hops, transform_dimensions):
    def __init__(self, num_hops, sigma, device, similarity_fun_name="inner product"):
        super().__init__()
        self.num_hops = num_hops
        self.sigma = sigma
        self.device = device
        self.sigmoid = nn.Sigmoid()
        self.similarity = Similarity(similarity_fun_name)

    def forward(self, x, edge_index):
        out = [torch.nn.functional.normalize(x, dim=1)]
        for k in range(self.num_hops):
            # Do we need to do a transform? I reckon we can use raw encoding and aggregate according to attention (and then the Classification module)
            # can handle how the aggregations get transformed
            # h = self.transforms[k](out[-1])
            h = out[-1]
            e_values = self.similarity(h[edge_index])
            # we have to use Sigmoid because if we use Softmax, removing an edge will change the weight of all other edges in the neighbourhood
            alpha_values = self.sigmoid(e_values)
            alpha = torch.sparse_coo_tensor(
                edge_index,
                alpha_values.reshape(edge_index.size(dim=1)),
                (x.size(dim=0), x.size(dim=0)),
                dtype=torch.float
            ).transpose(0, 1)

            aggr = torch.sparse.mm(alpha, h)
            # Might need to not use "transforms" and instead do raw aggregations like the original PMA
            # aggr = torch.mm(self.A, out[-1])
            # noised = aggr # TODO: add noise # Gaussian mechanism to guarantee DP for neighbourhood aggregation
            noised = aggr + torch.normal(torch.zeros(aggr.size()), std=self.sigma).to(self.device)
            normalized = torch.nn.functional.normalize(noised, dim=1)
            out.append(normalized)
        return torch.stack(out)
