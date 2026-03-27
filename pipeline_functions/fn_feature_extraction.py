# Feature extraction from .mat file for BE4999 SNN project, I used this to run .mat file data with 
# basic feature extraction (peak-to-peak, mean, Hjorth parameters) and save as .npy for SNN input. We can 
# add time window and features as needed! - Madalyn  3-4-2026

from scipy.io import loadmat, savemat
import numpy as np 


def extractFeatures(dataset, y, k=5, factor=5, save_filepath = None):
    """
    extracts features from an eeg dataset

    Inputs:
    - dataset: can be either a preloaded dataset of features or a filepath to a saved .mat dataset
    - y: labelling of each sample in X, can either be P300 binary labels for training or flash_ids for testing
    - k: how many trials to average together for class averaging
    - factor: the factor by which to downsample time series data
    - save_filepath: filepath for location save feature extracted dataset. If None, files are not saved

    Outputs:
    - X_norm: normalized time series data after feature extraction. Shape (trials/k, time/factor, channels)
    - y: y after trial averaging. Shape = original shape / k
    """
    X = None
    # Load .mat file -- Michael is adding to GitHub, but adjust filename as needed
    # if dataset is in an outside file
    if(isinstance(dataset, str)):
        mat = loadmat(dataset)
        X = mat['all_epochs']
        y = mat['labels']
    # if dataset is directly input
    else:
        X = dataset

    # Fix MATLAB->Python dimensions: (trials, channels, time) -- MATLAB used (channels, time, trials)
    X = np.transpose(X, (2, 0, 1))

    print("X shape:", X.shape)

    # averaging k trials together to reduce noise
    X, y = average_by_class(X, y, k=k)

    # time downsampling with decimation by average according to the factor
    X = decimation_by_avg(X, factor)

    # narrow to peak P300 window
    t_min, t_max = 105, 440
    t_step = 600/X.shape[2]
    step_min = round(t_min/t_step)
    step_max = round(t_max/t_step)

    X = X[:, :, step_min:step_max]

    # Normalize each trial per channel (0–1)
    eps = 1e-8

    ch_min = X.min(axis=(0, 2), keepdims=True)
    ch_max = X.max(axis=(0, 2), keepdims=True)

    X_norm = (X - ch_min) / (ch_max - ch_min + eps)

    print("Shape after feature extraction:", X_norm.shape)


    if save_filepath is not None:
        savemat(save_filepath,
            {'X': X_norm,
            'y': y})

    return X_norm, y

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

def decimation_by_avg(data, factor):
    """Function for replacing each sequence of previous factor samples with their average"""
    # as appears in Won et al., 2022
    # for example, frame [0, 800]ms -> 17samples (Krusienski et al., 2006)
    # data.shape = [trial, ch, time]
    ratio_dsample = factor
    n_trial, n_ch, n_frame = data.shape

    #print(n_frame)
    decimated_frame = int(np.floor(n_frame/ratio_dsample))
    #print(decimated_frame)

    # memory pre-allocation
    decimated_data = np.zeros((n_trial, n_ch, decimated_frame))
    #print(decimated_data.shape)

    for i in range(n_trial):
        for j in range(decimated_frame):
            cur_data = data[i, :, :]
            decimated_data[i, :, j] = np.mean(cur_data[:, j*ratio_dsample:(j+1)*ratio_dsample], axis=1)

    return decimated_data