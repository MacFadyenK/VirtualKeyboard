# Feature extraction from .mat file for BE4999 SNN project, I used this to run .mat file data with 
# basic feature extraction (peak-to-peak, mean, Hjorth parameters) and save as .npy for SNN input. We can 
# add time window and features as needed! - Madalyn  3-4-2026

from scipy.io import loadmat
import numpy as np 
import eeglib # for feature extraction, install via pip if needed (pip install eeglib)

# Load .mat file -- Michael is adding to GitHub, but adjust filename as needed
mat = loadmat("S17_Preprocessed_Epoch.mat")

# Extract datasets
target = mat["filtered_target_epochs"]
nontarget = mat["filtered_nontarget_epochs"]
times = mat["time"].squeeze()  # in ms

print("Times min:", times.min())
print("Times max:", times.max())
print("Times shape:", times.shape) 

# Fix MATLAB->Python dimensions: (trials, channels, time) -- MATLAB used (channels, time, trials)
target = np.transpose(target, (2, 0, 1))
nontarget = np.transpose(nontarget, (2, 0, 1))

# Combine trials
X = np.concatenate([target, nontarget], axis=0)
y = np.concatenate([np.ones(target.shape[0]), np.zeros(nontarget.shape[0])])

print("X shape:", X.shape)
print("y shape:", y.shape)

# Normalize each trial (0–1)
eps = 1e-8

trial_min = X.min(axis=(1, 2), keepdims=True)
trial_max = X.max(axis=(1, 2), keepdims=True)

X_norm = (X - trial_min) / (trial_max - trial_min + eps)

print("Shape after normalization:", X_norm.shape)

# Feature extraction per trial
# Example using EEGLib basic features: peak-to-peak, mean, Hjorth parameters
feature_list = []
for trial in X_norm:  # trial: (channels, time)
    trial_features = []
    for ch in trial:
        
        mean = np.mean(ch)
        ptp = np.ptp(ch) 

        # Hjorth manually --  activity, mobility, complexity
        diff1 = np.diff(ch)
        diff2 = np.diff(diff1)

        activity = np.var(ch)
        mobility = np.sqrt(np.var(diff1) / (activity + 1e-8))
        complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-8)) / (mobility + 1e-8)

        hjorth = (activity, mobility, complexity)

        trial_features.extend([ptp, mean] + list(hjorth))
    feature_list.append(trial_features)

features_array = np.array(feature_list)
print("Features array shape:", features_array.shape)

# Save files -- adjust path as needed 

np.save("X_norm.npy", X_norm)                 # (trials, channels, time)
np.save("X_features.npy", features_array)     # (trials, 160)
np.save("y.npy", y)                           # (trials,) , labels: 1 for target, 0 for nontarget
print("Saved X_norm.npy, X_features.npy, and y.npy")

#----------------------------------------------------------------------------------------------
# Saves:
# - X_norm.npy: normalized windowed EEG (numpy array, trials x channels x time) 
# - (commented out)X_flat.npy: flattened version (numpy array, trials x (channels*time))
# - X_features.npy: features (numpy array, trials x (channels*5))
#---------------------------------------------------------------------------------------------- 
# Notes:
# - X_norm is the time-series EEG for spike encoding / SNN input
# - X_features is an optional representation for ML baselines (shouldn't hurt to have both saved for flexibility)

#replaced variance features with peak-to-peak, var and activity are similar but ptp may capture more dynamic range 
# in the signal, keeping 5 features per channel

#if we ever need to flatten data for model input, we can do that after normalization, but we want the 3D shape 
#X_flat = X_norm.reshape(X_norm.shape[0], -1)  # 2D numpy array: (trials, channels*time)
#print("Flattened array for model input:", X_flat.shape)
#np.save("X_flat.npy", X_flat)


#can add time window selection before normalization if we want to focus on specific time ranges, 
# but since epochs are already time-constrained, we can skip this step for now. If needed, we can uncomment 
# and adjust the time window as necessary.

# Time window selection 
# tmin, tmax = 200, 500 # ms, expanded time window to capture more response, adjust as needed
#time_mask = (times >= tmin) & (times <= tmax)
#X_window = X[:, :, time_mask]
#print("Shape after time window selection:", X_window.shape)
