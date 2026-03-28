import numpy as np

def delta_encode(data, threshold=0.009, off_spike=True, padding=False):
    """
    Input: data shape = (trials, channels, time)
    Output: spikes shape = (trials, time, channels)
    """
    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 3:
        raise ValueError(f"Expected shape (trials, channels, time), got {data.shape}")

    if padding:
        prev = np.concatenate([data[:, :, 0:1], data[:, :, :-1]], axis=2)
    else:
        prev = np.concatenate([np.zeros_like(data[:, :, 0:1]), data[:, :, :-1]], axis=2)

    diff = data - prev

    if off_spike:
        spikes = np.zeros_like(diff, dtype=np.int8)
        spikes[diff >= threshold] = 1
        spikes[diff <= -threshold] = -1
    else:
        spikes = (diff >= threshold).astype(np.int8)

    return np.transpose(spikes, (0, 2, 1))
