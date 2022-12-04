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
from hyperparameters import *
from utils import compute_accuracy


def train(model, input, labels, train_mask, loss_fn, optimizer):
    model.train()

    # Compute posteriors and loss on training set
    posteriors = model(*input)
    loss = loss_fn(posteriors[train_mask], labels[train_mask])
    accuracy = compute_accuracy(posteriors[train_mask], labels[train_mask])

    # Do backpropogation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss), float(accuracy)


def build_encoder(input_dim, num_classes, train_loader):
    # Build encoder module
    encoder_dimensions = [input_dim, *encoder_hidden_dims, encoder_output_dim]
    encoder = MLPEncoder(encoder_dimensions).to(device)
    encoder_train = EncoderTrain(encoder, encoder_output_dim, num_classes).to(device)

    # Encoder pre-training
    print("Pretraining encoder...")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(encoder_train.parameters(), lr=1e-3)
    encoder_losses = []
    encoder_accuracies = []
    for t in range(1, encoder_training_iters+1):
        batch = next(iter(train_loader))
        loss, accuracy = train(encoder_train, [batch.x], batch.y, batch.train_mask, loss_fn, optimizer)
        if (t % 10) == 0:
            print("  Iter %3d: Loss = %0.3f --- Accuracy = %0.3f" % (t, loss, accuracy))
            encoder_losses.append(loss)
            encoder_accuracies.append(accuracy)
    # print("Encoder losses:", encoder_losses)
    # print("Encoder accuracies:", encoder_accuracies)

    # Freeze encoder gradients
    for param in encoder.parameters():
        param.requires_grad = False
    
    return encoder

def train_pmat(pmat, train_loader, encoder, num_examples, num_classes, pmat_epsilon, pmat_delta):
    # Create a temporary classification module to train PMAT
    base_dims = [encoder_output_dim, *classifier_base_hidden_dims, classifier_base_output_dim]
    head_dims = [(num_hops+1) * classifier_base_output_dim, *classifier_head_hidden_dims, num_classes]
    temp_classifier = MLPClassifier(num_hops, base_dims, head_dims).to(device)

    # Train PMAT 
    pmat_train = GAPBase(encoder, pmat, temp_classifier).to(device)

    loss_fn = nn.CrossEntropyLoss()
    # TODO: Not allowed dyanmic batch_size for DPSGD, our batches are edge-wise
    # so they should have fixed batch_size!
    optimizer = optim.DPAdam(
        l2_norm_clip=1.0,
        noise_multiplier=1.0,
        batch_size=batch_size,
        params=pmat_train.parameters(),
        lr=0.5e-1
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    edge_epochs = 0
    for t in range(1, pmat_training_iters+1):
        batch = next(iter(train_loader)).to(device)
        loss, accuracy = train(pmat_train, [batch.x, batch.edge_index], batch.y, batch.train_mask, loss_fn, optimizer)
        edge_epochs += batch_size / num_examples
        epsilon = analysis.moments_accountant(num_examples, batch_size, 1.0, edge_epochs, pmat_delta)
        if t % 50 == 0:
            print("  Iter %3d: Loss = %0.3f --- Accuracy = %0.3f --- (%0.3f, %f)-DP" %
                  (t, loss, accuracy, epsilon, pmat_delta))
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
        return PMA(num_hops, noise_scale).to(device)
    elif module_name == "pmwa":
        return PMWA(num_hops, noise_scale, device)
    elif module_name == "pmat":
        return PMAT(num_hops, encoder_output_dim, noise_scale)
    else:
        raise NotImplementedError("Unknown aggregation module type")


def main():
    # Load data
    train_data, test_data, num_classes = load_dataset("amazon", 0.1, device=device)
    train_loader = NeighborLoader(train_data, [-1] * num_hops, batch_size=batch_size, shuffle=True)
    test_loader = NeighborLoader(test_data, [-1] * num_hops, batch_size=batch_size)
    n, d = train_data.x.size()

    # Build and pre-train encoder module
    encoder = build_encoder(d, num_classes, train_loader)

    # Build aggregation module
    aggregation_module = build_aggregation_module(aggregation_module_name, 0)
    if aggregation_module_name.lower() == "pmat":
        print("Pre-training PMAT...")
        # TODO: Compute these based on hyperparameters
        pmat_epsilon = 4
        pmat_delta = 1e-5
        train_pmat(aggregation_module, train_loader, encoder, n, num_classes, pmat_epsilon, pmat_delta)

    # Build full model
    base_dims = [encoder_output_dim, *classifier_base_hidden_dims, classifier_base_output_dim]
    head_dims = [(num_hops+1) * classifier_base_output_dim, *classifier_head_hidden_dims, num_classes]
    classification_module = MLPClassifier(num_hops, base_dims, head_dims).to(device)
    model = GAPBase(encoder, aggregation_module, classification_module).to(device)

    # Train full model
    print("Training full model...")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model_losses = []
    model_accuracies = []
    for t in range(1, model_training_iters+1):
        batch = next(iter(train_loader)).to(device)
        loss, accuracy = train(model, [batch.x, batch.edge_index], batch.y, batch.train_mask, loss_fn, optimizer)
        if (t % 10) == 0:
            print("  Iter %3d: Loss = %0.3f --- Accuracy = %0.3f" % (t, loss, accuracy))
            model_losses.append(loss)
            model_accuracies.append(accuracy)
    # print("Model Losses:", model_losses)
    # print("Model Accuracies:", model_accuracies)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, stopping...")
