import numpy as np
from scipy.signal import butter, filtfilt
from scipy.io import loadmat
import h5py

def preprocess_training(file_path, save_path=None):
    """
    preprocesses EEG data for a training set

    Inputs:
    - file_path: path to the .mat file containing the training dataset
    - save_path: optional path to save the file in python format .npz. defaults to None to not save it

    Returns:
    - all_epochs: numpy array containing the preprocessed EEG signal data in shape (channels, time, signal)
    - labels: labels corresponding to the class of each signal in all_epochs
    """
    data = loadmat(file_path, squeeze_me=True, struct_as_record=False)

    t2 = data['train'][0]

    # lets us see where target stimulus and where non target stim is
    target_indices = np.where(t2.markers_target == 1)[0]
    nontarget_indices = np.where(t2.markers_target == 2)[0]

    # define parameters, sampling rate (512hz) and epoch length (600ms)
    samplingrate = t2.srate
    epoch_length = int(round(0.6 * samplingrate))

    # Design bandpass filter (0.5–15 Hz)
    b, a = butter(4, [0.5/(samplingrate/2), 15/(samplingrate/2)], btype='bandpass')

    # Use raw EEG data
    raw_data = t2.data

    # Pick first target stimulus
    firstargetstimulus = target_indices[0]

    pz_index = 12   # MATLAB 13 -> Python index 12

    time = np.arange(epoch_length) / samplingrate * 1000  # milliseconds

    # lets us create these arrays, channels, samples, and number of trials
    target_epochs = np.zeros((32, epoch_length, len(target_indices)))
    nontarget_epochs = np.zeros((32, epoch_length, len(nontarget_indices)))

    # Extract all target trials
    for i in range(len(target_indices)):
        firstargetstimulus = target_indices[i]
        target_epochs[:, :, i] = raw_data[:, firstargetstimulus:firstargetstimulus + epoch_length]

    # Extract all nontarget trials
    for i in range(len(nontarget_indices)):
        firstargetstimulus = nontarget_indices[i]
        nontarget_epochs[:, :, i] = raw_data[:, firstargetstimulus:firstargetstimulus + epoch_length]

    filteredepochs = np.zeros_like(target_epochs)
    filterednontargetepochs = np.zeros_like(nontarget_epochs)

    # Filter target trials
    for i in range(len(target_indices)):
        filteredepochs[:, :, i] = filtfilt(b, a, target_epochs[:, :, i], axis=1)

    # Filter nontarget trials
    for i in range(len(nontarget_indices)):
        filterednontargetepochs[:, :, i] = filtfilt(b, a, nontarget_epochs[:, :, i], axis=1)

    # Average across trials
    average_target = np.mean(filteredepochs, axis=2)
    average_nontarget = np.mean(filterednontargetepochs, axis=2)

    time = np.arange(epoch_length) / samplingrate * 1000

    # Compile dataset
    numberof_target = filteredepochs.shape[2]
    numberof_nontarget = filterednontargetepochs.shape[2]
    print(numberof_target)
    print(numberof_nontarget)

    labels = np.concatenate((np.ones(numberof_target, dtype=np.int64), np.zeros(numberof_nontarget, dtype=np.int64)))

    all_epochs = np.concatenate((filteredepochs, filterednontargetepochs), axis=2)

    # Save dataset as NPZ (Python format)
    if save_path is not None:
        np.savez(
            save_path,
            filtered_target_epochs=filteredepochs,
            filtered_nontarget_epochs=filterednontargetepochs,
            all_epochs=all_epochs,
            labels=labels,
            average_target=average_target,
            average_nontarget=average_nontarget,
            samplingrate=samplingrate,
            time=time,
            pz_index=pz_index
        )
        print("Preprocessed EEG dataset saved successfully.")

    return all_epochs, labels
