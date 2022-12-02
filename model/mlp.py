import torch.nn as nn

class MLP(nn.Sequential):
    def __init__(self, dimensions):
        layers = [nn.Flatten()]
        for i in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[i], dimensions[i+1]))
            layers.append(nn.SELU(inplace=True))
        super().__init__(*layers)

