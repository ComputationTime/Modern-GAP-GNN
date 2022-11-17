import torch.nn as nn


class GAPBase(nn.Module):
    encoder_module: nn.Module
    aggregation_module: nn.Module
    classification_module: nn.Module

    def __init__(self, encoder_module, aggregation_module, classification_module):
        self.encoder_module = encoder_module
        self.aggregation_module = aggregation_module
        self.classification_module = classification_module

    def forward(self, x, A):
        x = self.encoder_module(x)
        x = self.aggregation_module(x, A)
        x = self.classification_module(x)
        return x
