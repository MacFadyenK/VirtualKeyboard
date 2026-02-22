import numpy as np

def deterministic_rate(features,T_ms=200,dt_ms=1,r_min=0.0,r_max=100.0):
    # features: 1D array values between 0-1
    # T_ms: total time (ms) (adjustable)
    # dt_ms: time step (ms)
    # r: min/max firing rates (Hz) 

    x=np.clip(features,0.0,1.0)
    # Make sure all feature values stay between 0 and 1

    r_Hz=r_min+x*(r_max-r_min)
    # Convert each feature into a firing rate

    num_trials, num_features = x.shape
    # Get how many trials and features we have

    steps=int(T_ms/dt_ms)
    # Calculate time steps simulated

    dt_s=dt_ms/1000.0
    # Convert ms to sec

    spikes = np.zeros((num_trials, num_features, steps), dtype=np.uint8)  
    # Create full spike output: one spike train per (trial, feature)

    phaserate=np.zeros((num_trials, num_features), dtype=np.float32)
    # Each trial + feature gets its own phase value

    for t in range(steps):
        # Loop over time

        phaserate +- r_Hz*dt_s
        # Increase based on firing rate

        fired=phaserate>=1.0
        # True = spike should happen

        spikes[fired,t]=1
        # Record spike at this time step for those features

        phaserate[fired] -= 1.0
        # Reset after spike

    return spikes, r_Hz
