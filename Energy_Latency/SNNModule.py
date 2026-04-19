import snntorch as snn
from snntorch import utils, surrogate
import torch
import torch.nn as nn
import numpy as np
import time

# ============================================================
# SNN DEFINITION
# ============================================================

def createSNN(dim_inputs, hidden_layer, num_outputs=2,
              betas=[0.9, 0.9, 0.9],
              thresholds=[1, 1, 1],
              sigmoid_slope=10,
              p=0.1):
    return fcSNN(dim_inputs=dim_inputs,
                  hidden_layer=hidden_layer,
                  num_outputs=num_outputs,
                  betas=betas,
                  thresholds=thresholds,
                  sigmoid_slope=sigmoid_slope,
                  p=p)


class fcSNN(nn.Module):
    def __init__(self, dim_inputs, hidden_layer, num_outputs,
                 betas, thresholds, sigmoid_slope, p=0.1):
        super().__init__()

        self.fc1 = nn.Linear(dim_inputs, hidden_layer[0])
        self.lif1 = snn.Leaky(beta=betas[0],
                              threshold=thresholds[0],
                              init_hidden=True,
                              spike_grad=surrogate.fast_sigmoid(slope=sigmoid_slope))

        self.fc2 = nn.Linear(hidden_layer[0], hidden_layer[1])
        self.lif2 = snn.Leaky(beta=betas[1],
                              threshold=thresholds[1],
                              init_hidden=True,
                              spike_grad=surrogate.fast_sigmoid(slope=sigmoid_slope))

        self.fc3 = nn.Linear(hidden_layer[1], num_outputs)
        self.lif3 = snn.Leaky(beta=betas[2],
                              threshold=thresholds[2],
                              init_hidden=True,
                              output=True,
                              spike_grad=surrogate.fast_sigmoid(slope=sigmoid_slope))

        self.dp = nn.Dropout(p=p)

        # Weight init
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')

        # Bias init
        nn.init.uniform_(self.fc1.bias, a=-0.1, b=0.1)
        nn.init.uniform_(self.fc2.bias, a=-0.1, b=0.1)
        nn.init.uniform_(self.fc3.bias, a=-0.1, b=0.1)

    def forward(self, x, batch_first=False):
        # x shape should be (Time, Flashes, Channels)
        if batch_first:
            x = x.transpose(0, 1)

        utils.reset(self)
        spk_rec = []
        self.layer1_spikes = []
        self.layer2_spikes = []

        for step in range(x.size(0)):
            # x[step] is now (Flashes, Channels)
            cur1 = self.fc1(x[step])
            spk1 = self.dp(self.lif1(cur1))
            self.layer1_spikes.append(spk1)

            cur2 = self.fc2(spk1)
            spk2 = self.dp(self.lif2(cur2))
            self.layer2_spikes.append(spk2)

            cur3 = self.fc3(spk2)
            spk3, _ = self.lif3(cur3)
            spk_rec.append(spk3)

        return torch.stack(spk_rec, dim=0), None


# ============================================================
# SOP CALCULATION
# ============================================================

def count_snn_sops(model, input_spikes, T):
    h1 = model.fc1.out_features
    h2 = model.fc2.out_features
    out = model.fc3.out_features

    # Sum everything at once (Vectorized)
    s1_sum = torch.sum(torch.stack(model.layer1_spikes)).item()
    s2_sum = torch.sum(torch.stack(model.layer2_spikes)).item()

    sop1 = np.sum(input_spikes) * h1
    sop2 = s1_sum * h2
    sop3 = s2_sum * out

    total_sops = sop1 + sop2 + sop3
    mem_flops = (h1 + h2 + out) * T * 4
    
    return total_sops, mem_flops


# ============================================================
# SNN EVALUATION WITH METRICS
# ============================================================

def run_snn_with_metrics(model, spikes, power_watts=20.0, device="cpu"):
    # spikes shape: (Flashes, Time, Channels)
    spikes = np.asarray(spikes)
    n_epochs, T, C = spikes.shape
    
    model.to(device).eval()
    
    # Prepare input for Batch Processing: (Time, Batch, Channels)
    # This is MUCH faster than a loop
    x = torch.from_numpy(spikes).float().transpose(0, 1).to(device)

    with torch.no_grad():
        t_start = time.perf_counter()
        spkout, _mem = model(x) # One single pass for ALL flashes
        t_end = time.perf_counter()

    latency_snn = t_end - t_start

    # Now we calculate SOPs for the whole batch at once
    # Layer 1 & 2 spikes are stored in model.layer1_spikes as lists of (Batch, Neurons)
    s1_sum = torch.sum(torch.stack(model.layer1_spikes)).item()
    s2_sum = torch.sum(torch.stack(model.layer2_spikes)).item()

    h1, h2, out = model.fc1.out_features, model.fc2.out_features, model.fc3.out_features
    
    sop_input = np.sum(spikes) * h1
    sop_h1 = s1_sum * h2
    sop_h2 = s2_sum * out
    
    total_sops = sop_input + sop_h1 + sop_h2
    # Baseline maintenance for all neurons across all time and all flashes
    total_mem_flops = (h1 + h2 + out) * T * n_epochs * 4
    
    total_ops = total_sops + total_mem_flops
    energy = latency_snn * power_watts

    # Return structure matching your 7-variable unpack
    # spkout shape from model: (Time, Batch, Neurons)
    return (
        spkout,        # 1. spkout
        None,          # 2. lats (not needed for batch)
        latency_snn,   # 3. latency_snn
        None,          # 4. epoch_ops
        total_ops,     # 5. total_snn_ops
        energy,        # 6. snn_energy
        total_sops     # 7. total_sops
    )