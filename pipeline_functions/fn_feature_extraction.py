from scipy.io import loadmat, savemat
import numpy as np 


def extractFeatures(dataset, y, factor=5, t_min=105, t_max=440, norm_type='minmax', norm_factor=0.5, save_filepath = None):
    """
    extracts features from an eeg dataset

    Inputs:
    - dataset: can be either a preloaded dataset of features or a filepath to a saved .mat dataset
    - y: labelling of each sample in X, can either be P300 binary labels for training or flash_ids for testing
    - k: how many trials to average together for class averaging
    - factor: the factor by which to downsample time series data
    - t_min: minimum time point for feature extraction
    - t_max: maximum time point for feature extraction
    - norm_type: type of normalization to apply to features, either 'minmax' or 'std'
    - norm_factor: scaling factor to apply after standardization (only used if norm_type is 'std')
    - save_filepath: filepath for location save feature extracted dataset. If None, files are not saved

    Outputs:
    - X_norm: normalized time series data after feature extraction. Shape (trials/k, time/factor, channels)
    - y: y after trial averaging. Shape = original shape / k
    """
    X = None
    # Load .mat file 
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

    # print("X shape:", X.shape)

    # time downsampling with decimation by average according to the factor
    X = decimation_by_avg(X, factor)

    # narrow to peak P300 window
    t_step = 600/X.shape[2]
    step_min = round(t_min/t_step)
    step_max = round(t_max/t_step)

    X = X[:, :, step_min:step_max]

    # Normalize each trial per channel (0–1)
    eps = 1e-8

    if norm_type == 'std':
        ch_mean = X.mean(axis=(0, 2), keepdims=True)
        ch_std = X.std(axis=(0, 2), keepdims=True)

        X_norm = (X - ch_mean) / (ch_std + eps)
        # scale
        X_norm = X_norm * norm_factor
    elif norm_type == 'minmax':
        ch_min = X.min(axis=(0, 2), keepdims=True)
        ch_max = X.max(axis=(0, 2), keepdims=True)

        X_norm = (X - ch_min) / (ch_max - ch_min + eps)

    # print("Shape after feature extraction:", X_norm.shape)


    if save_filepath is not None:
        savemat(save_filepath,
            {'X': X_norm,
            'y': y})

    return X_norm, y


def decimation_by_avg(data, factor):
    """Function for replacing each sequence of previous factor samples with their average"""
    # altered from what appears in Won et al., 2022 to be more efficient
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