# Feature extraction from .mat file for BE4999 SNN project, I used this to run .mat file data with 
# basic feature extraction (peak-to-peak, mean, Hjorth parameters) and save as .npy for SNN input. We can 
# add time window and features as needed! - Madalyn  3-4-2026

from scipy.io import loadmat
import numpy as np 


def extractFeatures(dataset, times, save_filepath = None):
    """
    extracts features from an eeg dataset

    Inputs:
    - dataset: can be either a preloaded dataset of features or a filepath to a saved .mat dataset
    - save_filepath: filepath for location save feature extracted dataset. If None, files are not saved

    Outputs:
    - X_norm: normalized time series data before feature extraction
    - features_array: feature extracted numpy dataset
    """
    X = None
    # Load .mat file -- Michael is adding to GitHub, but adjust filename as needed
    # if dataset is in an outside file
    if(isinstance(dataset, str)):
        mat = loadmat(dataset)
        X = mat['all_epochs']
    # if dataset is directly input
    else:
        X = dataset

    # Fix MATLAB->Python dimensions: (trials, channels, time) -- MATLAB used (channels, time, trials)
    X = np.transpose(X, (2, 0, 1))

    print("X shape:", X.shape)

    # Normalize each trial (0–1)
    eps = 1e-8

    trial_min = X.min(axis=(1, 2), keepdims=True)
    trial_max = X.max(axis=(1, 2), keepdims=True)

    X_norm = (X - trial_min) / (trial_max - trial_min + eps)

    print("Shape after normalization:", X_norm.shape)

    # P300-specific feature extraction per trial/channel
    # Features per channel:
    # 1. peak amplitude
    # 2. peak latency (ms)
    # 3. mean amplitude
    # 4. area under curve
    # 5. standard deviation
    feature_list = []

    for trial in X:  # trial: (channels, time)
        trial_features = []

        for ch in trial:
            max_amp = np.max(ch)
            max_latency = times[np.argmax(ch)]   # uses full epoch time axis
            mean_amp = np.mean(ch)
            auc = np.sum(ch)
            std_amp = np.std(ch)

            trial_features.extend([max_amp, max_latency, mean_amp, auc, std_amp])

        feature_list.append(trial_features)

    features_array = np.array(feature_list)
    print("Features array shape:", features_array.shape)

    # Normalize feature array
    features_min = features_array.min(axis=0, keepdims=True)
    features_max = features_array.max(axis=0, keepdims=True)

    fe_norm = (features_array - features_min) / (features_max - features_min + eps)

    # Save tensor and features -- currently to desktop, adjust path as needed 


    if save_filepath is not None:
        np.save(save_filepath + "X_norm.npy", X_norm)                 # (trials, channels, time)
        np.save(save_filepath + "X_features.npy", features_array)     # (trials, 160)
        print("Saved X_norm.npy, X_features.npy")

    # (nTrials x nFeatures) reshape for SNN input
    tensor_reshaped = X_norm.reshape(X_norm.shape[0], -1)
    print("Reshaped tensor for SNN input:", tensor_reshaped.shape)

    return X_norm, features_array, fe_norm

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