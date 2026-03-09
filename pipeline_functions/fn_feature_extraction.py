# Feature extraction from .mat file for BE4999 SNN project, I used this to run .mat file data with 
# basic feature extraction (peak-to-peak, mean, Hjorth parameters) and save as .npy for SNN input. We can 
# add time window and features as needed! - Madalyn  3-4-2026

from scipy.io import loadmat
import numpy as np 
import eeglib # for feature extraction, install via pip if needed (pip install eeglib)

def extractFeatures(dataset, save_filepath = None):
    """
    extracts features from an eeg dataset

    Inputs:
    - dataset: can be either a preloaded dataset or a filepath to a saved .mat dataset
    - save_filepath: filepath for location save feature extracted dataset. If None, files are not saved

    Outputs:
    - X_norm: normalized time series data before feature extraction
    - features_array: feature extracted numpy dataset
    - y: labels for each sample
    """
    # Load .mat file -- Michael is adding to GitHub, but adjust filename as needed
    # if dataset is in an outside file
    if(isinstance(dataset, str)):
        mat = loadmat(dataset)
    # if dataset is directly input
    else:
        mat = dataset

    # Extract datasets
    target = mat["filtered_target_epochs"]
    nontarget = mat["filtered_nontarget_epochs"]
    times = mat["time"].squeeze()  # in seconds

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
    # Example using EEGLib basic features: variance, mean, Hjorth parameters
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

    # Save tensor and features -- currently to desktop, adjust path as needed 


    if save_filepath is not None:
        np.save(save_filepath + "X_norm.npy", X_norm)                 # (trials, channels, time)
        np.save(save_filepath + "X_features.npy", features_array)     # (trials, 160)
        np.save(save_filepath + "y.npy", y)                           # (trials,) , labels: 1 for target, 0 for nontarget
        print("Saved X_norm.npy, X_features.npy, and y.npy")

    # (nTrials x nFeatures) reshape for SNN input
    tensor_reshaped = X_norm.reshape(X_norm.shape[0], -1)
    print("Reshaped tensor for SNN input:", tensor_reshaped.shape)

    return X_norm, features_array, y

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