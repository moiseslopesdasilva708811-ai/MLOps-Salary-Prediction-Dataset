import torch
import torch.nn as nn

class NeuralNetworkMLP(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetworkMLP, self).__init__()
        # Architecture ReLU
        self.camadas = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.camadas(x)