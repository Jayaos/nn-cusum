import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation="relu"):
        """
        hidden_dims: int or list[int]
            - int: same width for all hidden layers
            - list: width of each hidden layer
        """
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        self.activation_name = activation

        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1]) for i in range(len(dims) - 1)
        ])

        print(f"Using MLP with {len(hidden_dims)} hidden layers, dims = {hidden_dims}")

    def _act(self, x):
        if self.activation_name == "relu":
            return F.relu(x)
        elif self.activation_name == "softplus":
            return F.softplus(x, beta=20.0, threshold=20.0)
        elif self.activation_name == "tanh":
            return torch.tanh(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")

    def forward(self, x, last_activation=None):
        for layer in self.layers[:-1]:
            x = self._act(layer(x))
        x = self.layers[-1](x)

        if last_activation:
            return last_activation(x)
        else:
            return x
