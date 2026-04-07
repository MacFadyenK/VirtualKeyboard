import torch
import torch.nn as nn


def createANN(dim_inputs, hidden_layer, num_outputs=2, dropout=0.2):
    """
    Wrapper that builds a fully connected ANN for EEG classification.

    Inputs:
    - dim_inputs: flattened input dimension size
    - hidden_layer: list like [256, 128]
    - num_outputs: number of output classes. Default = 2
    - dropout: dropout probability applied after hidden layers. Default = 0.2

    Returns:
    - fcANN model
    """
    return fcANN(dim_inputs=dim_inputs, hidden_layer=hidden_layer, num_outputs=num_outputs, dropout=dropout)


class fcANN(nn.Module):
    def __init__(self, dim_inputs, hidden_layer, num_outputs=2, dropout=0.2):
        super().__init__()

        if len(hidden_layer) < 2:
            raise ValueError("hidden_layer must contain at least two values, e.g. [256, 128]")

        self.flatten = nn.Flatten(start_dim=1)

        self.fc1 = nn.Linear(dim_inputs, hidden_layer[0])
        self.bn1 = nn.BatchNorm1d(hidden_layer[0])
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_layer[0], hidden_layer[1])
        self.bn2 = nn.BatchNorm1d(hidden_layer[1])
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_layer[1], num_outputs)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='linear')

        nn.init.uniform_(self.fc1.bias, a=-0.1, b=0.1)
        nn.init.uniform_(self.fc2.bias, a=-0.1, b=0.1)
        nn.init.uniform_(self.fc3.bias, a=-0.1, b=0.1)

    def forward(self, x):
        """
        Forward pass for ANN.

        Expected input shapes:
        - (batch, channels, time)
        - (batch, features)

        Returns:
        - logits of shape (batch, num_outputs)
        """
        if x.ndim > 2:
            x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        logits = self.fc3(x)
        return logits
