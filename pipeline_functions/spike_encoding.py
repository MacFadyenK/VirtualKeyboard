import numpy as np

def deterministic_rate(features,T_ms=200,dt_ms=1,r_min=50.0,r_max=150.0):
    """
    deterministic rate encoding of a dataset

    Inputs:
    - features: array values between 0-1
    - T_ms: total time (ms) (adjustable)
    - dt_ms: time step (ms)
    - r_min: min firing rate (Hz) 
    - r_max: max firing rate (Hz)

    Output:
    - spikes: spike encoded feature dataset of shape (trials, features, time_steps)
    - r_Hz: each feature's firing rate (Hz) shape (trials, features)
    """

    x = np.clip(features,0.0,1.0).astype(np.float32)
    # Make sure all feature values stay between 0 and 1

    r_Hz = r_min+x*(r_max-r_min)
    # Convert each feature into a firing rate

    steps = int(T_ms/dt_ms)
    # Calculate time steps simulated

    dt_s = dt_ms/1000.0
    # Convert ms to sec

    spikes = np.zeros(x.shape + (steps,), dtype=np.uint8)  
    # Create full spike output

    phaserate = np.zeros_like(r_Hz, dtype=np.float32)
    # Each trial + feature gets its own phase value

    for t in range(steps):
        # Loop over time

        phaserate += r_Hz*dt_s
        # Increase based on firing rate

        fired = phaserate >= 1.0
        # True = spike should happen

        spikes[..., t] = fired.astype(np.uint8)
        # Record spike at this time step for those features

        phaserate[fired] -= 1.0
        # Reset after spike

    return spikes, r_Hz