import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader

from datasets.datasets import load_dataset
from model.encoder_module import MLPEncoder
from model.encoder_module import EncoderTrain
from model.aggregation_module import PrivateMultihopAggregation
from model.classification_module import MLPClassifier
from model.model import GAPBase
from hyperparameters import *
from utils import compute_accuracy


def train(model: nn.Module, input, labels, train_mask, loss_fn, optimizer):
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

def main():
    torch.manual_seed(seed)

    # Load data
    train_data, test_data, num_classes = load_dataset("amazon", 0.1)
    train_x, train_y, train_edge_index = train_data.x, train_data.y, train_data.edge_index
    train_mask = train_data.train_mask
    train_loader = NeighborLoader(
        train_data,
        num_neighbors=[-1] * num_hops,
        batch_size=batch_size,
        shuffle=True
    )
    n_train, d = train_x.size()

    # Build encoder module
    encoder_dimensions = [d, *encoder_hidden_dims, encoder_output_dim]
    encoder = MLPEncoder(encoder_dimensions)
    encoder_train = EncoderTrain(encoder, encoder_output_dim, num_classes)

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

    # Build full model
    aggregation_module = PrivateMultihopAggregation(num_hops, 0)
    base_dims = [encoder_output_dim, *classifier_base_hidden_dims, classifier_base_output_dim]
    head_dims = [(num_hops+1) * classifier_base_output_dim, *classifier_head_hidden_dims, num_classes]
    classification_module = MLPClassifier(num_hops, base_dims, head_dims)
    model = GAPBase(encoder, aggregation_module, classification_module)

    # Train full model
    print("Training full model...")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model_losses = []
    model_accuracies = []
    for t in range(1, model_training_iters+1):
        batch = next(iter(train_loader))
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
