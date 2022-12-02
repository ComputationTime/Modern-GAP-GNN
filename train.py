import torch
import torch.nn as nn

from datasets.datasets import load_dataset
from datasets.preprocessing import get_adjacency_matrix
from model.encoder_module import MLPEncoder
from model.encoder_module import EncoderTrain
from model.aggregation_module import PrivateMultihopAggregation
from model.classification_module import MLPClassifier
from model.model import GAPBase
from hyperparameters import *


def train(model: nn.Module, input_data, labels, loss_fn, optimizer):
    model.train()

    # Compute predictions and loss on training set
    prediction = model(*input_data)
    loss = loss_fn(prediction, labels)
    accuracy = (prediction.argmax(dim=1) == labels).sum() / labels.size(dim=0)
    # loss = loss_fn(prediction[data.train_mask], data.y[data.train_mask])

    # Do backpropogation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss), float(accuracy)


if __name__ == "__main__":
    torch.manual_seed(seed)

    # Load data
    train_data, test_data, num_classes = load_dataset("amazon", 0.1)
    train_A = get_adjacency_matrix(train_data)
    test_A = get_adjacency_matrix(test_data)
    n_train, d = train_data.x.size()

    # Build encoder module
    encoder_dimensions = [d, *encoder_hidden_dims, encoder_output_dim]
    encoder = MLPEncoder(encoder_dimensions)
    encoder_train = EncoderTrain(encoder, encoder_output_dim, num_classes)

    # Encoder pre-training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(encoder_train.parameters(), lr=1e-3)
    encoder_losses = []
    encoder_accuracies = []
    for t in range(encoder_epochs):
        loss, accuracy = train(encoder_train, [train_data.x], train_data.y, loss_fn, optimizer)
        if ((t+1) % 10) == 0:
            encoder_losses.append(loss)
            encoder_accuracies.append(accuracy)
    print("Encoder losses:", encoder_losses)
    print("Encoder accuracies:", encoder_accuracies)

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
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model_losses = []
    model_accuracies = []
    for t in range(model_epochs):
        loss, accuracy = train(model, [train_data.x, train_A], train_data.y, loss_fn, optimizer)
        if ((t+1) % 10) == 0:
            model_losses.append(loss)
            model_accuracies.append(accuracy)
    print("Model Losses:", model_losses)
    print("Model Accuracies:", model_accuracies)
