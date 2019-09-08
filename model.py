import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.nn.functional as f


class DQNetwork(nn.Module):
    """Actor (Policy) Model used for DQNetworks"""
    def __init__(self, state_size: int, action_size: int, seed: int = 42, fc1_units: int = 64, fc2_units: int = 32):
        """Initialize the model parameters and build a model with 2 hidden layers.

           Args:
                state_size (int): Dimension of each state
                action_size (int): Dimension of each action
                seed (int): Random seed
                fc1_units (int): Number of nodes in first hidden layer
                fc2_units (int): Number of nodes in second hidden layer
        """
        super(DQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """
            Forward propagation of input.
            Args:
                state (PyTorch model): The observed state used to predict actions
        """
        x = relu(self.fc1(state))
        x = relu(self.fc2(x))
        return self.fc3(x)


class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model used for Dueling QNetworks"""

    def __init__(self, state_size: int, action_size: int, seed: int = 42, fc1_units: int = 64, fc2_units: int = 32):
        """Initialize the model parameters and build a model with 2 hidden layers.

           Args:
                state_size (int): Dimension of each state
                action_size (int): Dimension of each action
                seed (int): Random seed
                fc1_units (int): Number of nodes in first hidden layer
                fc2_units (int): Number of nodes in second hidden layer
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        self.action_size = action_size
        self.value_function_fc = nn.Linear(fc2_units, 1)
        self.advantage_function_fc = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """
            Forward propagation of input.
            Args:
                state (PyTorch model): The observed state used to predict actions
        """
        x = f.relu(self.fc1(state))
        x = f.relu(self.fc2(x))

        value_function = self.value_function_fc(x)
        advantage_function = self.advantage_function_fc(x)

        return value_function + advantage_function - \
            advantage_function.mean(1).unsqueeze(1).expand(x.size(0), self.action_size) / self.action_size

