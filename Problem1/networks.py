import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleProbabilityMLP(nn.Module):
    def __init__(self, input_dimensions, hidden_layers_size):
        super(SimpleProbabilityMLP, self).__init__()
        self.fc1 = nn.Linear(input_dimensions, hidden_layers_size)
        self.fc2 = nn.Linear(hidden_layers_size, hidden_layers_size)
        self.fc3 = nn.Linear(hidden_layers_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)
