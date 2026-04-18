import torch
import torch.nn as nn
import numpy as np
import time

# ============================================================
# ANN DEFINITION
# ============================================================

def createANN(dim_inputs, hidden_layer, num_outputs=2, dropout=0.2):
    """
    Wrapper that builds a fully connected ANN for EEG classification.
    """
    return fcANN(dim_inputs=dim_inputs, 
                 hidden_layer=hidden_layer, 
                 num_outputs=num_outputs, 
                 dropout=dropout)


class fcANN(nn.Module):
    def __init__(self, dim_inputs, hidden_layer, num_outputs=2, dropout=0.2):
        super().__init__()

        if len(hidden_layer) < 2:
            raise ValueError("hidden_layer must contain at least two values, e.g. [256, 128]")

        self.flatten = nn.Flatten(start_dim=1)

        # Layer 1
        self.fc1 = nn.Linear(dim_inputs, hidden_layer[0])
        self.bn1 = nn.BatchNorm1d(hidden_layer[0])
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        # Layer 2
        self.fc2 = nn.Linear(hidden_layer[0], hidden_layer[1])
        self.bn2 = nn.BatchNorm1d(hidden_layer[1])
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        # Layer 3 (Output)
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
        Input x shape: (batch, channels, time) or (batch, features)
        Returns: logits (batch, num_outputs)
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


# ============================================================
# ANN METRIC CALCULATION (FLOPs)
# ============================================================

def count_ann_flops(model, input_shape):
    """
    Calculates total FLOPs for the forward pass.
    Linear FLOPs ≈ 2 * in_features * out_features (for Multiply-Accumulate)
    """
    batch_size = input_shape[0]
    # Calculate input features after flattening
    in_dim = np.prod(input_shape[1:])
    
    h1 = model.fc1.out_features
    h2 = model.fc2.out_features
    out = model.fc3.out_features

    # FC Layers
    flops_fc1 = 2 * in_dim * h1
    flops_fc2 = 2 * h1 * h2
    flops_fc3 = 2 * h2 * out
    
    # Overhead: BatchNorm (4 ops/neuron) + ReLU (1 op/neuron)
    # Total overhead is small, but included for completeness
    flops_overhead = (h1 + h2) * 5 
    
    total_flops = (flops_fc1 + flops_fc2 + flops_fc3 + flops_overhead) * batch_size
    return total_flops


# ============================================================
# ANN EVALUATION RUNNER
# ============================================================

def run_ann_with_metrics(model, data, power_watts=20.0, device="cpu"):
    """
    Benchmarks the ANN and returns classification results and metrics.
    """
    model.to(device).eval()
    
    # Ensure data is a Torch tensor
    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data).float()
    
    data = data.to(device)
    
    # Warm up (optional, for more stable latency measurements)
    _ = model(data[:1]) 
    
    with torch.no_grad():
        t_start = time.perf_counter()
        logits = model(data)
        t_end = time.perf_counter()
    
    latency = t_end - t_start
    total_flops = count_ann_flops(model, data.shape)
    energy = latency * power_watts
    
    return (
        logits,       # 1. spkout (logits for ANN)
        None,         # 2. lats (not applicable)
        latency,      # 3. latency_ann
        0,            # 4. epoch_ops (not applicable)
        total_flops,  # 5. total_ann_ops
        energy,       # 6. ann_energy
        total_flops   # 7. total_ann_flops (re-using flops here)
    )