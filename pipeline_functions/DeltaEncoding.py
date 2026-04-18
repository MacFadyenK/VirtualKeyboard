import numpy as np
import time
import matplotlib.pyplot as plt
import scipy


# ============================================================
# ORIGINAL DELTA ENCODING (UNCHANGED)
# ============================================================

def delta_encode(data, threshold=0.09, off_spike=True, padding=False):
    """
    Input: data shape = (trials, channels, time)
    Output: spikes shape = (trials, time, channels)
    """
    data = np.asarray(data, dtype=np.float32)
    time_enc = time.time()
    flops_enc = 0

    if data.ndim != 3:
        raise ValueError(f"Expected shape (trials, channels, time), got {data.shape}")

    if padding:
        prev = np.concatenate([data[:, :, 0:1], data[:, :, :-1]], axis=2)
    else:
        prev = np.concatenate([np.zeros_like(data[:, :, 0:1]), data[:, :, :-1]], axis=2)

    # Subtraction: 1 op per element
    diff = data - prev
    flops_enc += data.size

    if off_spike:
        spikes = np.zeros_like(diff, dtype=np.int8)
        # Two comparisons: 2 ops per element
        spikes[diff >= threshold] = 1
        spikes[diff <= -threshold] = -1
        flops_enc += diff.size * 2
    else:
        # One comparison: 1 op per element
        spikes = (diff >= threshold).astype(np.int8)
        flops_enc += diff.size

    latency_enc = time.time() - time_enc

    return np.transpose(spikes, (0, 2, 1)), latency_enc, flops_enc


# ============================================================
# DELTA ENCODING WITH FLOPs, LATENCY, ENERGY
# ============================================================

def delta_encode_with_metrics(all_epochs, threshold=0.009, off_spike=True, padding=False, power_watts=20.0):
    """
    all_epochs: shape (channels, time, epochs)

    Returns:
        spikes_out
        per_epoch_latencies
        total_latency
        per_epoch_flops
        total_flops
        per_epoch_energy
        total_energy
    """

    n_channels, n_time, n_epochs = all_epochs.shape

    per_epoch_latencies = []
    per_epoch_flops = []
    per_epoch_energy = []
    spikes_out = []

    # FLOPs per element (Nick-style)
    FLOPS_PER_ELEMENT = 3 if off_spike else 2

    for ep in range(n_epochs):

        epoch_data = all_epochs[:, :, ep]           # (channels, time)
        epoch_data = epoch_data[None, :, :]         # reshape to (1, channels, time)

        # -------------------------
        # LATENCY START
        # -------------------------
        t_start = time.perf_counter()

        spikes = delta_encode(
            epoch_data,
            threshold=threshold,
            off_spike=off_spike,
            padding=padding
        )

        # -------------------------
        # LATENCY END
        # -------------------------
        t_end = time.perf_counter()
        latency = t_end - t_start
        per_epoch_latencies.append(latency)

        # -------------------------
        # FLOP COUNTING
        # -------------------------
        num_elements = epoch_data.size
        FLOPS = num_elements * FLOPS_PER_ELEMENT
        per_epoch_flops.append(FLOPS)

        # -------------------------
        # ENERGY
        # -------------------------
        per_epoch_energy.append(latency * power_watts)

        spikes_out.append(spikes[0])

    # Convert lists to arrays
    spikes_out = np.stack(spikes_out, axis=0)
    per_epoch_latencies = np.array(per_epoch_latencies)
    per_epoch_flops = np.array(per_epoch_flops)
    per_epoch_energy = np.array(per_epoch_energy)

    total_latency = per_epoch_latencies.sum()
    total_flops = per_epoch_flops.sum()
    total_energy = per_epoch_energy.sum()

    return (spikes_out,
            per_epoch_latencies,
            total_latency,
            per_epoch_flops,
            total_flops,
            per_epoch_energy,
            total_energy)
