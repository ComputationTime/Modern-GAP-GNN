import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import Sequential, GCNConv
from torch_geometric.data import Data
from pyvacy import optim, analysis
import sys
device = "cuda" if torch.cuda.is_available() else "cpu"

DEBUG = False
def debug_print(*args): DEBUG and print(*args)

# Edge level DP
delta = 1e-5
K_hop = 1
batch_size = 256

class MLP(nn.Module):
  # e.g. dimensions = [50,40,30,20]
    def __init__(self, dimensions):
        super().__init__()
        self.flatten = nn.Flatten()
        layers = []
        for i in range(len(dimensions)-1):
          layers.append(nn.Linear(dimensions[i], dimensions[i+1]))
          layers.append(nn.SELU(inplace=True))

        self.linear_selu_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_selu_stack(x)
        return logits
class AggregationModule(nn.Module):
  edge_index = None

  def __init__(self):
        super().__init__()

class PMA(AggregationModule):
    # A - adjacency matrix     TODO: this should not be given to the module itself, it should access it in training (or from the graph dataset)
    # num_hops - the number of hops covered by this GNN
    def __init__(self, num_hops, sigma):
        super().__init__()
        self.num_hops = num_hops
        self.sigma = sigma

    def forward(self, x):
        # TEMP SOLUTION
        if AggregationModule.edge_index is None:
          raise RuntimeError("Set AggregationModule.edge_index [TEMP SOLUTION] before running")
        edge_index = AggregationModule.edge_index
        A = get_adjacency_matrix(edge_index, x.size(dim=0))
        out = [torch.nn.functional.normalize(x, dim=1)]
        for k in range(self.num_hops):
            aggr = torch.mm(torch.transpose(A, 0, 1), out[-1].to(device))
            noised = aggr + torch.normal(torch.zeros(aggr.size()), std=self.sigma).to(device)
            normalized = torch.nn.functional.normalize(noised, dim=1)
            out.append(normalized)
        return torch.stack(out)
        # return torch.nn.functional.normalize(x, dim=1)
class PMAT(AggregationModule):
    # def __init__(self, num_hops, transform_dimensions):
    def __init__(self, num_hops, encoding_dimensions, sigma):
        super().__init__()
        self.num_hops = num_hops
        self.sigma = sigma
        self.sigmoid = nn.Sigmoid()
        # self.transforms = nn.ModuleList()
        self.attentions = nn.ModuleList()
        for i in range(num_hops):
          # self.transforms.append(nn.Linear(*transform_dimensions)) # Only 1 layer transformation
          # self.attentions.append(MLP([2*transform_dimensions[-1], 1])) # Attention mechanism takes 2 encodings and outputs 1 weight
          # TODO: Figure out if we want a transformer?
          self.attentions.append(MLP([2*encoding_dimensions, 1]))

    def forward(self, x):
        # TEMP SOLUTION
        if AggregationModule.edge_index is None:
          raise RuntimeError("Set AggregationModule.edge_index [TEMP SOLUTION] before running")
        edge_index = AggregationModule.edge_index
        out = [torch.nn.functional.normalize(x, dim=1)]
        for k in range(self.num_hops):
            # Do we need to do a transform? I reckon we can use raw encoding and aggregate according to attention (and then the Classification module)
            # can handle how the aggregations get transformed
            # h = self.transforms[k](out[-1])
            h = out[-1]
            e_values = self.attentions[k](h[edge_index.T].reshape(edge_index.size(dim=1), 2*h.size(dim=1))) # DPSGD to guarantee DP attention training
            # we have to use Sigmoid because if we use Softmax, removing an edge will change the weight of all other edges in the neighbourhood
            alpha_values = self.sigmoid(e_values)
            alpha = torch.sparse_coo_tensor(edge_index,
                                            alpha_values.reshape(edge_index.size(dim=1)),
                                            (x.size(dim=0), x.size(dim=0)),
                                            dtype=torch.float).transpose(0, 1)

            aggr = torch.sparse.mm(alpha, h)
            # Might need to not use "transforms" and instead do raw aggregations like the original PMA
            # aggr = torch.mm(self.A, out[-1]) 
            # noised = aggr # TODO: add noise # Gaussian mechanism to guarantee DP for neighbourhood aggregation
            noised = aggr + torch.normal(torch.zeros(aggr.size()), std=self.sigma).to(device)
            normalized = torch.nn.functional.normalize(noised, dim=1)
            out.append(normalized)
        return torch.stack(out)
class Similarity(nn.Module):
    def __init__(self, function_name: str):
        super().__init__()
        if function_name == "inner product":
            self.fun = lambda x : torch.sum(x[0, :]*x[1, :], dim = 1)
        else:
            raise NotImplementedError("Not a valid similarity function")
            
    def forward(self, x):
        return self.fun(x)

class PMWA(AggregationModule):
    # def __init__(self, num_hops, transform_dimensions):
    def __init__(self, num_hops, sigma, similarity_fun_name = "inner product"):
        super().__init__()
        self.num_hops = num_hops
        self.sigma = sigma
        self.sigmoid = nn.Sigmoid()
        # self.transforms = nn.ModuleList()
        self.similarity = Similarity(similarity_fun_name)

    def forward(self, x):
        # TEMP SOLUTION
        if AggregationModule.edge_index is None:
          raise RuntimeError("Set AggregationModule.edge_index [TEMP SOLUTION] before running")
        edge_index = AggregationModule.edge_index
        out = [torch.nn.functional.normalize(x, dim=1)]
        for k in range(self.num_hops):
            # Do we need to do a transform? I reckon we can use raw encoding and aggregate according to attention (and then the Classification module)
            # can handle how the aggregations get transformed
            # h = self.transforms[k](out[-1])
            h = out[-1]
            e_values = self.similarity(h[edge_index])
            # we have to use Sigmoid because if we use Softmax, removing an edge will change the weight of all other edges in the neighbourhood
            alpha_values = self.sigmoid(e_values)
            alpha = torch.sparse_coo_tensor(edge_index,
                                            alpha_values.reshape(edge_index.size(dim=1)),
                                            (x.size(dim=0), x.size(dim=0)),
                                            dtype=torch.float).transpose(0, 1)

            aggr = torch.sparse.mm(alpha, h)
            # Might need to not use "transforms" and instead do raw aggregations like the original PMA
            # aggr = torch.mm(self.A, out[-1]) 
            # noised = aggr # TODO: add noise # Gaussian mechanism to guarantee DP for neighbourhood aggregation
            noised = aggr + torch.normal(torch.zeros(aggr.size()), std=self.sigma).to(device)
            normalized = torch.nn.functional.normalize(noised, dim=1)
            out.append(normalized)
        return torch.stack(out)
        
class Classification(nn.Module):
    # num_hops - the number of hops covered by this GNN
    # encoder_dimensions - the MLP dimensions of each base MLP
    # head_dimensions - the dimensions of the head MLP
    def __init__(self, num_hops, encoder_dimensions, head_dimensions):
        super().__init__()
        self.base_mlps = nn.ModuleList()
        self.num_hops = num_hops
        if encoder_dimensions:
          for i in range(num_hops+1):
              self.base_mlps.append(MLP(encoder_dimensions))
        self.head_mlp = MLP(head_dimensions) # TODO: should this be softmax? I think we add a softmax for classification tasks. We can test if it works better
    
    def forward(self, cache):
        # forward through bases
        out = []
        for i in range(self.num_hops+1):
          if self.base_mlps:
            encoding = self.base_mlps[i](cache[i,:,:])
            out.append(encoding) # add corresponding encoding
          else:
            out.append(cache[i, :, :])
        # combine (use concatenation)
        combined_x = torch.cat(out, dim=1)
        # forward through head
        return self.head_mlp(combined_x)



class GAP(nn.Module):
  # encoder - pretrained encoder module
  # pma - PMA module
  # classification - classification module
  def __init__(self, encoder, pma, classification): # TODO: decide whether we should recieve the models as parameters
    super().__init__()
    self.encoder = encoder
    self.encoder.requires_grad=False
    self.pma = pma
    self.classification = classification

  def forward(self, x):
    if AggregationModule.edge_index is not None:
      # initial node encoding
      x_encoded = self.encoder(x)
      # aggregation module
      cache = self.pma(x_encoded)
    else:
      cache = x
    # classification
    return self.classification(cache) 

# this method partitions based on nodes (so edges between splits are not used)
def train_test_split(dataset, test_ratio):
    X, y, edge_index= dataset.x, dataset.y, dataset.edge_index
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
    edge_index_test = edge_index[:, torch.logical_and(*~mask[edge_index])] - train_slice

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

def prepare_dataset(dataset, threshold):
    X, y, edge_index = dataset.x, dataset.y, dataset.edge_index

    # remove labels with less examples than threshold
    index_map = torch.zeros(y.size())
    included_classes = y.unique(return_counts=True)[1] >= threshold
    filter = included_classes[y]
    # remap labels (i.e. if they were 0-8 and we remove 4 labels, new labels should be between 0 and 4)
    label_mapping = torch.zeros(included_classes.size(dim=0), dtype=torch.long)
    label_mapping[included_classes] = torch.arange(torch.count_nonzero(included_classes))

    y = label_mapping[y[filter]].to(torch.long)
    X = X[filter]

    # remove edges that had their nodes removed
    edge_index = filter_edge_index(edge_index, filter)

    return Data(x=X, y=y, edge_index=edge_index)

# make sparse adjacency matrix, A
def get_adjacency_matrix(edge_index, num_nodes):
    values = torch.ones(edge_index.size(dim=1), dtype = torch.int).to(device)
    A = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes), dtype=torch.float)
    return A

def standardization(train_dataset, test_dataset):
    X = train_dataset.x
    means = X.mean(dim=0, keepdim=True)
    stds = X.std(dim=0, keepdim=True)
    X_train = (X - means) / stds
    X_test = (test_dataset.x - means) / stds
    return Data(x=X_train, y=train_dataset.y, edge_index=train_dataset.edge_index), Data(x=X_test, y=test_dataset.y, edge_index=test_dataset.edge_index)

def add_self_edges(dataset):
    X = dataset.x
    self_edges = torch.stack((torch.arange(X.size(dim=0)), torch.arange(X.size(dim=0))))
    edge_index = torch.cat((dataset.edge_index, self_edges), dim=1)
    return Data(x=X, y=dataset.y, edge_index=edge_index)


# train
def train(batch, model, loss_fn, optimizer):
  model.train()
  X, y = batch.x.to(device), batch.y.to(device)
  if batch.edge_index is not None:
    AggregationModule.edge_index = batch.edge_index.to(device)
  else:
    AggregationModule.edge_index = None
  # compute prediction error
  pred = model(X)
  loss = loss_fn(pred[:batch.batch_size], y[:batch.batch_size])
  # backpropagation
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  torch.cuda.empty_cache()

# test batch
def batch_test(batch, split, model, loss_fn, wordy=False):
    size = batch.batch_size
    model.eval()
    test_loss, correct = 0, 0
    with torch.inference_mode():
        X, y = batch.x.to(device), batch.y.to(device)
        AggregationModule.edge_index = batch.edge_index.to(device)
        pred = model(X)
        test_loss = loss_fn(pred[:batch.batch_size], y[:batch.batch_size]).item()
        correct = (pred[:batch.batch_size].argmax(1) == y[:batch.batch_size]).type(torch.float).sum().item() / size
    if wordy:
      print(f"{split.title()} Error: \n Accuracy: {(100*correct):>0.1f}%, Loss: {test_loss:>8f}")
    return test_loss, correct

# test
def test(loader, split, model, loss_fn):
  size = len(loader)
  model.eval()
  test_loss, correct = 0, 0
  for data in loader:
    # HACK: 
    if type(data) == list:
      x, y = data
      x = x.transpose(0, 1) # dim 0 (hop) ; dim 1 (n) ; dim 2 (d)
      batch = Data(x=x, y=y, batch_size=y.size(0))
    else:
      batch = data
    batch_loss, batch_correct = batch_test(batch, split, model, loss_fn)
    test_loss += batch_loss
    correct += batch_correct
  correct /= size
  test_loss /= size
  print(f"{split.title()} Error: \n Avg Accuracy: {(100*correct):>0.1f}%, Avg Loss: {test_loss:>8f}")


def compute_aggregation_sigma(model_name, epsilon, pmat_epsilon, iterations):
    global delta, K_hop
    if model_name == "pmat":
        # agg_sigma = 1 / np.max(np.roots([K_hop/2, (3/np.sqrt(2))*np.sqrt(2*K_hop*np.log(1/delta)), pmat_epsilon - epsilon]))
        agg_eps = epsilon - pmat_epsilon
        agg_sigma = 1 / np.max(np.roots([iterations*K_hop/2, np.sqrt(2*iterations*K_hop*np.log(1/(delta/2))), -agg_eps]))
    else:
        agg_eps = epsilon
        agg_sigma = 1 / np.max(np.roots([K_hop/2, np.sqrt(2*K_hop*np.log(1/delta)), -agg_eps]))
    return agg_sigma






















def main(model_name, dataset_name, eps):
  global device
  global delta 
  global K_hop
  global batch_size

  print(model_name, dataset_name, eps)

  if dataset_name == "reddit":
    dimensions = [602, 300, 60]
    classification_dims = [60, 20]
    from torch_geometric.datasets import Reddit
    dataset = Reddit('./rd')[0]
    # prepare dataset by removing classes that have less than 1000 examples
    dataset = prepare_dataset(dataset, 10000)
    num_classes = torch.unique(dataset.y).size(dim=0)
    train_dataset, test_dataset = train_test_split(dataset, 0.2)
    from torch_geometric.loader import NeighborLoader

    X_train, y_train, edge_index_train = train_dataset.x, train_dataset.y, train_dataset.edge_index
    X_test, y_test, edge_index_test = test_dataset.x, test_dataset.y, test_dataset.edge_index

    # using large number like 10,000 so that all neighbours are sampled 
    # I don't like how it samples, so I'm just gonna sample everything
    train_loader_encoder = NeighborLoader(train_dataset, num_neighbors=[X_train.size(dim=0)]*K_hop, 
                                          batch_size=batch_size, shuffle=True)
    train_loader = NeighborLoader(train_dataset, num_neighbors=[X_train.size(dim=0)]*K_hop, 
                                  batch_size=batch_size)
    test_loader = NeighborLoader(test_dataset, num_neighbors=[X_test.size(dim=0)]*K_hop, 
                                 batch_size=batch_size)

  elif dataset_name == "amazon":
    dimensions = [767, 300, 60]
    classification_dims = [60, 20]
    from torch_geometric.datasets import Amazon
    dataset = Amazon('./amzn', "Computers")[0]
    # prepare dataset by removing classes that have less than 1000 examples
    dataset = prepare_dataset(dataset, 1000)
    num_classes = torch.unique(dataset.y).size(dim=0)
    train_dataset, test_dataset = train_test_split(dataset, 0.2)
    from torch_geometric.loader import NeighborLoader

    X_train, y_train, edge_index_train = train_dataset.x, train_dataset.y, train_dataset.edge_index
    X_test, y_test, edge_index_test = test_dataset.x, test_dataset.y, test_dataset.edge_index

    # using large number like 10,000 so that all neighbours are sampled 
    # I don't like how it samples, so I'm just gonna sample everything
    train_loader_encoder = NeighborLoader(train_dataset, num_neighbors=[X_train.size(dim=0)]*K_hop, 
                                          batch_size=batch_size, shuffle=True)
    train_loader = NeighborLoader(train_dataset, num_neighbors=[X_train.size(dim=0)]*K_hop, 
                                  batch_size=batch_size)
    test_loader = NeighborLoader(test_dataset, num_neighbors=[X_test.size(dim=0)]*K_hop, 
                                 batch_size=batch_size)

  elif dataset_name == "facebook":
    dimensions = [128, 64, 32]
    classification_dims = [32, 16]
    from torch_geometric.datasets import FacebookPagePage
    dataset = FacebookPagePage('./fb')[0]
    num_classes = torch.unique(dataset.y).size(dim=0)
    train_dataset, test_dataset = train_test_split(dataset, 0.2)
    from torch_geometric.loader import NeighborLoader

    X_train, y_train, edge_index_train = train_dataset.x, train_dataset.y, train_dataset.edge_index
    X_test, y_test, edge_index_test = test_dataset.x, test_dataset.y, test_dataset.edge_index

    # using large number like 10,000 so that all neighbours are sampled 
    # I don't like how it samples, so I'm just gonna sample everything
    train_loader_encoder = NeighborLoader(train_dataset, num_neighbors=[X_train.size(dim=0)]*K_hop, 
                                          batch_size=batch_size, shuffle=True)
    train_loader = NeighborLoader(train_dataset, num_neighbors=[X_train.size(dim=0)]*K_hop, 
                                  batch_size=batch_size)
    test_loader = NeighborLoader(test_dataset, num_neighbors=[X_test.size(dim=0)]*K_hop, 
                                 batch_size=batch_size)

  pmat_opt_iterations = 1000
  noise_multiplier = 1.3
  pmat_epsilon = analysis.moments_accountant(X_train.size(dim=0), batch_size, noise_multiplier, (batch_size/X_train.size(dim=0))*pmat_opt_iterations, delta/2)
  if model_name == "pmat" and pmat_epsilon > eps:
    return
  agg_sigma = compute_aggregation_sigma(model_name, eps, pmat_epsilon, pmat_opt_iterations)
  print(f"Epsilon: {eps:>0.2f}, Sigma: {agg_sigma:>0.2f}")


  encoder_model = nn.Sequential(
      MLP(dimensions),
      nn.Linear(dimensions[-1], num_classes),
      nn.Softmax(dim=1)
  )

  encoder_model = encoder_model.to(device)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(encoder_model.parameters(), lr=1e-3)

  for t in range(1000):
      batch = next(iter(train_loader_encoder))
      train(batch, encoder_model, loss_fn, optimizer)
  test(test_loader, "TEST", encoder_model, loss_fn)

  encoder = encoder_model[0]
  encoder.requires_grad=False
  for param in encoder.parameters():
    param.requires_grad = False

  if model_name == "pma":
    agg_func = PMA(K_hop, agg_sigma)
    model = GAP(encoder,
                agg_func,
                Classification(K_hop, classification_dims, [(K_hop+1)*classification_dims[-1], num_classes]))
  elif model_name == "pmwa":
    agg_func = PMWA(K_hop, agg_sigma)
    model = GAP(encoder,
                agg_func,
                Classification(K_hop, classification_dims, [(K_hop+1)*classification_dims[-1], num_classes]))
  elif model_name == "pmat":
    # Get alpha
    # alpha = np.sqrt((2*agg_sigma*agg_sigma*np.log(1/delta)) / K_hop) + 1
    # print("alpha", alpha)
    # Create a temporary classification module to train PMAT
    base_dims = []
    head_dims = [(K_hop+1) * dimensions[-1], num_classes]
    temp_classifier = Classification(K_hop, base_dims, head_dims).to(device)
    pmat_train = nn.Sequential(encoder, PMAT(K_hop, dimensions[-1], agg_sigma), temp_classifier).to(device)

    loss_fn = nn.CrossEntropyLoss()
    # TODO: Not allowed dyanmic batch_size for DPSGD, our batches are edge-wise
    # so they should have fixed batch_size!
    optimizer = optim.DPAdam(
        l2_norm_clip=1.0,
        noise_multiplier=noise_multiplier,
        batch_size=batch_size,
        params=pmat_train.parameters(),
        lr=0.5e-1
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    
    edge_epochs = 0
    for t in range(pmat_opt_iterations):
        batch = next(iter(train_loader_encoder))
        train(batch, pmat_train, loss_fn, optimizer)
        edge_epochs += batch_size / X_train.size(dim=0)
        # epsilon = analysis.moments_accountant(X_train.size(dim=0), batch_size, noise_multiplier, edge_epochs, delta/2)
        
        # rdp = analysis.rdp_accountant.compute_rdp(batch_size/X_train.size(dim=0), noise_multiplier, t+1, [alpha])
        # epsilon, _, _ = analysis.rdp_accountant.get_privacy_spent([alpha], rdp, target_delta=delta)
      
        # if (t + 1) % 100 == 0:
          # print("Epoch:", edge_epochs)
          # batch_test(next(iter(train_loader_encoder)), "TRAIN", pmat_train, loss_fn, True)
          # print("Optimizer Achieves ({:>0.1f}, {})-DP".format(epsilon, delta))
          # print("LR:", scheduler.get_last_lr()[0])
        scheduler.step()
    # debug_print("Trained PMAT in %d iterations" % t)
    test(test_loader, "TEST", pmat_train, loss_fn)

    pmat = pmat_train[1]
    pmat.requires_grad = False
    for param in pmat.parameters():
      param.requires_grad = False

    agg_func = pmat
    model = GAP(encoder,
                agg_func,
                Classification(K_hop, classification_dims, [(K_hop+1)*classification_dims[-1], num_classes]))

  model = model.to(device)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.6)

  cache_x = []
  cache_y = []
  for batch in train_loader:
    X, y = batch.x.to(device), batch.y.to(device)
    AggregationModule.edge_index = batch.edge_index.to(device)
    cache_x.append(agg_func(encoder(X))[:, :batch.batch_size, :])
    cache_y.append(y[:batch.batch_size])
  cache_x = torch.cat(cache_x, dim=1) # dim 0 (hop) ; dim 1 (n) ; dim 2 (d)
  cache_x = cache_x.transpose(0, 1) # dim 0 (n) ; dim 1 (hop) ; dim 2 (d)
  cache_y = torch.cat(cache_y, dim=0)
  # print(f"cache_x ({cache_x.size()}):", cache_x[:5])

  from torch.utils.data import DataLoader, TensorDataset
  cache = TensorDataset(cache_x, cache_y)
  train_loader = DataLoader(cache, batch_size=batch_size, shuffle=True)

  for t in range(1000):
    x, y = next(iter(train_loader))
    x = x.transpose(0, 1) # dim 0 (hop) ; dim 1 (n) ; dim 2 (d)
    batch = Data(x=x, y=y, batch_size=y.size(0))
    train(batch, model, loss_fn, optimizer)
    scheduler.step()
  test(test_loader, "TEST", model, loss_fn)













if __name__ == "__main__":
  try:
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]))
  except KeyboardInterrupt:
    print("Received KeyboardInterrupt... cancelling.")

