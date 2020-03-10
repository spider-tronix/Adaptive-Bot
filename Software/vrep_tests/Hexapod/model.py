import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, num_layers, hidden_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layers = nn.ModuleList()
        in_size = state_size
        for i in range(num_layers):
            self.layers.append(nn.Linear(in_size,hidden_size[i]))
            in_size = hidden_size[i]
        self.layers.append(nn.Linear(in_size, action_size))

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for layer in self.layers:
            x = F.relu(layer(x))
        return x
