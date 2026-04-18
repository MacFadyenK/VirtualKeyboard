# Feature extraction from .mat file for BE4999 SNN project, I used this to run .mat file data with 
# basic feature extraction (peak-to-peak, mean, Hjorth parameters) and save as .npy for SNN input. We can 
# add time window and features as needed! - Madalyn  3-4-2026

from scipy.io import loadmat, savemat
import numpy as np 
import time


def extractFeatures(dataset, y, factor=5, t_min=200, t_max=400, norm_type= 'std', norm_factor=1.0, save_filepath = None):
    time_feat = time.time()
    flops_feat = 0
    X = None

    if(isinstance(dataset, str)):
        mat = loadmat(r"C:\Users\maryf\OneDrive\Documents\BE4999\0_Archive\s17_preprocessed.mat")
        X = mat['all_epochs']
        y = mat['labels']
    else:
        X = dataset

    X = np.transpose(X, (2, 0, 1))

    # Decimation FLOPs
    flops_feat += X.size 
    X = decimation_by_avg(X, factor)

    t_step = 600/X.shape[2]
    step_min = round(t_min/t_step)
    step_max = round(t_max/t_step)

    X = X[:, :, step_min:step_max]

    eps = 1e-8

    if norm_type == 'std':
        ch_mean = X.mean(axis=(0, 2), keepdims=True)
        flops_feat += X.size
        
        ch_std = X.std(axis=(0, 2), keepdims=True)
        flops_feat += X.size * 4

        X_norm = (X - ch_mean) / (ch_std + eps)
        flops_feat += X.size * 3
        
        X_norm = X_norm * 0.5
        flops_feat += X_norm.size
        
    elif norm_type == 'minmax':
        ch_min = X.min(axis=(0, 2), keepdims=True)
        ch_max = X.max(axis=(0, 2), keepdims=True)
        flops_feat += X.size * 2

        X_norm = (X - ch_min) / (ch_max - ch_min + eps)
        flops_feat += X.size * 3

    if save_filepath is not None:
        savemat(save_filepath, {'X': X_norm, 'y': y})

    latency_feat = time.time() - time_feat

    return X_norm, y, latency_feat, flops_feat

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

def average_by_class(X, y, k=5):
    """ averages k number of trials in X of the same class in y together"""
    X_avg, y_avg = [], []
    
    for cls in np.unique(y):
        X_cls = X[y == cls]
        for i in range(0, len(X_cls) - k + 1, k):
            X_avg.append(X_cls[i:i+k].mean(axis=0))
            y_avg.append(cls)

    return np.stack(X_avg), np.array(y_avg)

from collections import defaultdict

def average_by_class_streaming(X, y, k=5):
    """Average every k samples per class in the order they are encountered."""
    buffers = defaultdict(list)   # holds ongoing samples per class
    X_avg, y_avg = [], []

    for xi, yi in zip(X, y):
        buffers[yi].append(xi)

        if len(buffers[yi]) == k:
            X_avg.append(np.mean(buffers[yi], axis=0))
            y_avg.append(yi)
            buffers[yi].clear()  # reset for next chunk

    return np.stack(X_avg), np.array(y_avg)

def decimation_by_avg(data, factor):
    """Function for replacing each sequence of previous factor samples with their average"""
    # altered from what appears in Won et al., 2022
    # data.shape = [trial, ch, time]
    n_trial, n_ch, n_frame = data.shape

    #print(n_frame)
    decimated_frame = n_frame//factor
    # #print(decimated_frame)

    # trim data so its divisible by factor
    data = data[:, :, :decimated_frame * factor]
    # print(data.shape)

    # decimate by average
    decimated_data = data.reshape(n_trial, n_ch, decimated_frame, factor).mean(axis=3)

    return decimated_data

if __name__ == "__main__":
    print("Loading preprocessed EEG → extracting features...\n")

    # ---------------------------------------------------------
    # 1. Load preprocessed EEG data
    # ---------------------------------------------------------
    data = loadmat("s17_preprocessed.mat")

    all_epochs = data["all_epochs"]      # shape: (channels, time, epochs)
    labels = data["labels"].squeeze()    # shape: (epochs,)
    time_vec = data["time"].squeeze()    # shape: (time,)

    n_channels, n_samples, n_epochs = all_epochs.shape
    print(f"Loaded {n_epochs} epochs with {n_channels} channels")

    # ---------------------------------------------------------
    # 2. Extract FEATURES from each epoch
    # ---------------------------------------------------------
    # Example: mean amplitude per channel per epoch
    # Output shape: (epochs, channels)
    features = all_epochs.mean(axis=1).T.astype(np.float32)

    print(f"Feature matrix shape: {features.shape}  (epochs × channels)")

    # ---------------------------------------------------------
    # 3. Normalize features (optional)
    # ---------------------------------------------------------
    f_min = features.min()
    f_max = features.max()

    if f_max > f_min:
        features_norm = (features - f_min) / (f_max - f_min)
    else:
        features_norm = np.zeros_like(features)

    print("Features normalized to 0–1")

    # ---------------------------------------------------------
    # 4. Save features for spike encoding script
    # ---------------------------------------------------------
    savemat("s17_features.mat", {
        "features": features,
        "features_norm": features_norm,
        "labels": labels
    })

    print("\nSaved extracted features to s17_features.mat")
    print("Feature extraction complete.")
