import torch.nn as nn

from .mlp import MLP


def MLPEncoder(dimensions):
    return MLP(dimensions)


def EncoderTrain(encoder, output_dim, num_classes):
    return nn.Sequential(encoder, nn.Linear(output_dim, num_classes), nn.Softmax(dim=1))
