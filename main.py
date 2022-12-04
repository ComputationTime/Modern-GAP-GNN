import sys

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
from pyvacy import analysis, optim

from datasets.datasets import load_dataset
from model.encoder_module import MLPEncoder
from model.encoder_module import EncoderTrain
from model.aggregation_module import PMA, PMAT, PMWA
from model.classification_module import MLPClassifier
from model.model import GAPBase
from hyperparameters import config
from utils import compute_accuracy

DEBUG = False
def debug_print(*argv): DEBUG and print(*argv)


def train(model, inputs, labels, batch_size, loss_fn, optimizer):
    model.train()

    # Compute posteriors and loss on training set
    posteriors = model(*inputs)
    loss = loss_fn(posteriors[:batch_size], labels[:batch_size])
    accuracy = compute_accuracy(posteriors[:batch_size], labels[:batch_size])

    # Do backpropogation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.cuda.empty_cache()

    return float(loss), float(accuracy)

def test(model, inputs, labels, batch_size, loss_fn):
    with torch.inference_mode():
        # Compute posteriors and loss on test set
        posteriors = model(*inputs)
        loss = loss_fn(posteriors[:batch_size], labels[:batch_size])
        accuracy = compute_accuracy(posteriors[:batch_size], labels[:batch_size])
    return float(loss), float(accuracy)


def test_encoder(encoder, loader, loss_fn):
    size = len(loader)
    encoder.eval()
    test_loss, accuracy = 0, 0
    for batch in loader:
        batch_loss, batch_accuracy = test(encoder, [batch.x], batch.y, batch.batch_size, loss_fn)
        test_loss += batch_loss
        accuracy += batch_accuracy
    accuracy /= size
    test_loss /= size
    print(f"Encoder: Loss = {test_loss:>8f} --- Accuracy: {(100*accuracy):>0.1f}%")


def test_model(model, loader, loss_fn):
    size = len(loader)
    model.eval()
    test_loss, accuracy = 0, 0
    for batch in loader:
        batch_loss, batch_accuracy = test(model, [batch.x, batch.edge_index], batch.y, batch.batch_size, loss_fn)
        test_loss += batch_loss
        accuracy += batch_accuracy
    accuracy /= size
    test_loss /= size
    print(f"Model: Loss = {test_loss:>8f} --- Accuracy: {(100*accuracy):>0.1f}%")


def build_encoder(input_dim, num_classes, train_loader, test_loader):
    # Build encoder module
    encoder_dimensions = [input_dim, *config.encoder_hidden_dims, config.encoder_output_dim]
    encoder = MLPEncoder(encoder_dimensions).to(config.device)
    encoder_train = EncoderTrain(encoder, config.encoder_output_dim, num_classes).to(config.device)

    # Encoder pre-training
    debug_print("Pretraining encoder...")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(encoder_train.parameters(), lr=1e-3)
    encoder_losses = []
    encoder_accuracies = []
    for t in range(1, config.encoder_training_iters+1):
        batch = next(iter(train_loader))
        loss, accuracy = train(encoder_train, [batch.x], batch.y, batch.batch_size, loss_fn, optimizer)
        if (t % 10) == 0:
            debug_print("  Iter %3d: Loss = %0.3f --- Accuracy = %0.3f" % (t, loss, accuracy))
            encoder_losses.append(loss)
            encoder_accuracies.append(accuracy)

    # Freeze encoder gradients
    for param in encoder.parameters():
        param.requires_grad = False

    test_encoder(encoder_train, test_loader, loss_fn)
    
    return encoder

def train_pmat(pmat, train_loader, encoder, num_examples, num_classes, pmat_epsilon, pmat_delta):
    # Create a temporary classification module to train PMAT
    base_dims = [
        config.encoder_output_dim, 
        *config.classifier_base_hidden_dims,
        config.classifier_base_output_dim
    ]
    head_dims = [
        (config.num_hops+1) * config.classifier_base_output_dim, 
        *config.classifier_head_hidden_dims,
        num_classes
    ]
    temp_classifier = MLPClassifier(config.num_hops, base_dims, head_dims).to(config.device)

    # Train PMAT 
    pmat_train = GAPBase(encoder, pmat, temp_classifier).to(config.device)

    loss_fn = nn.CrossEntropyLoss()
    # TODO: Not allowed dyanmic batch_size for DPSGD, our batches are edge-wise
    # so they should have fixed batch_size!
    optimizer = optim.DPAdam(
        l2_norm_clip=1.0,
        noise_multiplier=1.0,
        batch_size=config.batch_size,
        params=pmat_train.parameters(),
        lr=0.5e-1
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    edge_epochs = 0
    for t in range(1, config.pmat_training_iters+1):
        batch = next(iter(train_loader)).to(config.device)
        loss, accuracy = train(pmat_train, [batch.x, batch.edge_index], batch.y, batch.batch_size, loss_fn, optimizer)
        edge_epochs += config.batch_size / num_examples
        epsilon = analysis.moments_accountant(num_examples, config.batch_size, 1.0, edge_epochs, pmat_delta)
        if t % 50 == 0:
            debug_print(
                "  Iter %3d: Loss = %0.3f --- Accuracy = %0.3f --- (%0.3f, %f)-DP" %
                (t, loss, accuracy, epsilon, pmat_delta)
            )
        scheduler.step()
        if epsilon >= pmat_epsilon:
            break

    # Freeze PMAT parameters
    for param in pmat.parameters():
        param.requires_grad = False


def build_aggregation_module(module_name, noise_scale):
    """Creates the requested aggregation module and a function which pre-trains
    the module"""
    module_name = module_name.lower()
    if module_name == "pma":
        return PMA(config.num_hops, noise_scale).to(config.device)
    elif module_name == "pmwa":
        return PMWA(config.num_hops, noise_scale, config.device)
    elif module_name == "pmat":
        return PMAT(config.num_hops, config.encoder_output_dim, noise_scale)
    else:
        raise NotImplementedError("Unknown aggregation module type")


def compute_aggregation_sigma():
    num_hops, epsilon, delta = config.num_hops, config.epsilon, config.delta
    if config.aggregation_module_name == "pmat":
        agg_eps = epsilon * 0.8 - np.log(config.delta)/(config.alpha - 1)
    else:
        agg_eps = epsilon
    agg_sigma = 1 / np.max(np.roots([num_hops/2, np.sqrt(2*num_hops*np.log(1/delta)), -agg_eps]))
    return agg_sigma


def main():
    torch.manual_seed(config.seed)

    # Compute noise_scale based on epsilon, delta, and alpha
    noise_scale = compute_aggregation_sigma()
    print(config.aggregation_module_name, config.dataset_name, config.epsilon) # OUTPUT
    print(f"Epsilon: {config.epsilon:>0.2f}, Sigma: {noise_scale:>0.3f}") # OUTPUT

    # Load data
    train_data, test_data, num_classes = load_dataset(config.dataset_name, 0.1)
    train_data.to(config.device)
    test_data.to(config.device)
    train_loader = NeighborLoader(train_data, [-1] * config.num_hops, batch_size=config.batch_size, shuffle=True)
    test_loader = NeighborLoader(test_data, [-1] * config.num_hops, batch_size=config.batch_size)
    n, d = train_data.x.size()

    # Build and pre-train encoder module
    encoder = build_encoder(d, num_classes, train_loader, test_loader)

    # Build aggregation module
    aggregation_module = build_aggregation_module(config.aggregation_module_name, noise_scale)
    if config.aggregation_module_name.lower() == "pmat":
        debug_print("Pre-training PMAT...")
        train_pmat(aggregation_module, train_loader, encoder, n, num_classes, config.epsilon, config.delta)

    # Build full model
    base_dims = [
        config.encoder_output_dim,
        *config.classifier_base_hidden_dims,
        config.classifier_base_output_dim
    ]
    head_dims = [
        (config.num_hops+1) * config.classifier_base_output_dim,
        *config.classifier_head_hidden_dims,
        num_classes
    ]
    classification_module = MLPClassifier(config.num_hops, base_dims, head_dims).to(config.device)
    model = GAPBase(encoder, aggregation_module, classification_module).to(config.device)

    # Train full model
    debug_print("Training full model...")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.6)
    model_losses = []
    model_accuracies = []
    for t in range(1, config.model_training_iters+1):
        batch = next(iter(train_loader)).to(config.device)
        loss, accuracy = train(model, [batch.x, batch.edge_index], batch.y, batch.batch_size, loss_fn, optimizer)
        # scheduler.step()
        if (t % 10) == 0:
            debug_print("  Iter %3d: Loss = %0.3f --- Accuracy = %0.3f" % (t, loss, accuracy))
            model_losses.append(loss)
            model_accuracies.append(accuracy)
    test_model(model, test_loader, loss_fn)
    print()


if __name__ == "__main__":
    # Parse arguments in the form "--[hyperparameter_name] hyperparameter value"
    args = ' ' + ' '.join(sys.argv[1:])
    args = args.split(' --')[1:]
    for arg in args:
        arg = arg.split(' ', 1)
        if len(arg) == 1:
            print(f"No value supplied for option --{arg[0]}, ignoring.")
        elif hasattr(config, arg[0]):
            if type(getattr(config, arg[0])) == str:
                setattr(config, arg[0], arg[1].strip())
            else:
                exec(f"config.{arg[0]} = {arg[1]}")
        else:
            print(f"Unknown config option --{arg[0]}, ignoring.")

    try:
        main()
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, stopping...")
