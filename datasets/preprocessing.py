import torch
from torch_geometric.data import Data


# this method partitions based on nodes (so edges between splits are not used)
def train_test_split(dataset, test_ratio):
    X, y, edge_index = dataset.x, dataset.y, dataset.edge_index
    shuffle_ordering = torch.randperm(X.size(dim=0))

    edge_mapping = torch.zeros(X.size(dim=0), dtype=torch.long)
    edge_mapping[shuffle_ordering] = torch.arange(X.size(dim=0))

    X = X[shuffle_ordering]
    y = y[shuffle_ordering]
    edge_index = edge_mapping[edge_index]

    mask = torch.zeros(X.size(dim=0), dtype=torch.bool)
    train_slice = int((1-test_ratio)*X.size(dim=0))
    mask[:train_slice] = True

    X_train = X[mask]
    X_test = X[~mask]

    y_train = y[mask]
    y_test = y[~mask]

    edge_index_train = edge_index[:, torch.logical_and(*mask[edge_index])]
    edge_index_test = edge_index[:, torch.logical_and(
        *~mask[edge_index])] - train_slice

    return Data(x=X_train, y=y_train, edge_index=edge_index_train), \
        Data(x=X_test, y=y_test, edge_index=edge_index_test)


# returns filtered edge index, first removes edges that have removed src or dst nodes, then shifts indices of remained src/dst nodes
def filter_edge_index(edge_index, filter):
    node_indices = torch.arange(filter.size(dim=0))[filter]
    edge_mapping = torch.zeros(filter.size(dim=0), dtype=torch.long)
    edge_mapping[node_indices] = torch.arange(node_indices.size(dim=0))

    edge_index = edge_index.to(torch.long)
    edge_filter = torch.logical_and(*filter[edge_index])
    return edge_mapping[edge_index[:, edge_filter]]


def remove_infrequent_classes(dataset, threshold):
    X, y, edge_index = dataset.x, dataset.y, dataset.edge_index

    # remove labels with less examples than threshold
    included_classes = y.bincount() >= threshold
    filter = included_classes[y]
    X = X[filter]
    y = y[filter]

    # remap labels (e.g. if they were 0-8 and we remove 4 labels, new labels should be between 0 and 4)
    label_mapping = torch.zeros(included_classes.size(dim=0), dtype=torch.long)
    label_mapping[included_classes] = torch.arange(included_classes.sum())
    y = label_mapping[y]

    # remove edges that had their nodes removed
    edge_index = filter_edge_index(edge_index, filter)

    return Data(x=X, y=y, edge_index=edge_index)


def get_adjacency_matrix(dataset):
    edge_index = dataset.edge_index
    n = dataset.x.size(dim=0)
    n_edges = edge_index.size(dim=1)

    # make sparse adjacency matrix, A
    values = torch.ones(n_edges, dtype=torch.int)
    A = torch.sparse_coo_tensor(edge_index, values, (n, n), dtype=torch.float)

    return A
