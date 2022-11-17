import torch.nn as nn

class MLP(nn.Sequential):
    def __init__(self, dimensions, activation_layer = nn.ReLU):
        layers = []
        for i in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[i], dimensions[i+1]))
            layers.append(activation_layer())
        super().__init__(*layers)

