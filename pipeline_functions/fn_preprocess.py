import numpy as np
from scipy.signal import butter, filtfilt, detrend
from scipy.io import loadmat, savemat
import mat73

def preprocess_training(file_path, save_path=None):
    """
    preprocesses EEG data for a training set

    Inputs:
    - file_path: path to the .mat file containing the training dataset
    - save_path: optional path to save the file as a .mat. defaults to None to not save it

    Returns:
    - all_epochs: numpy array containing the preprocessed EEG signal data in shape (channels, time, signal)
    - labels: labels corresponding to the class of each signal in all_epochs
    - time: the time each epoch takes up
    """
    data = mat73.loadmat(file_path)

    labels=[]
    all_epochs=[]

    
    # define parameters, sampling rate (512hz) and epoch length (600ms)
    samplingrate = float(data['train'][0]['srate'])

    baseline_length = round(0.2 * samplingrate)
    epoch_length = round(0.6 * samplingrate)

    # channels for use
    selected_channels = np.array([31,32,12,13,19,14,18,16]) - 1

    # Design bandpass filter (0.5–15 Hz)
    b, a = butter(4, [0.5/(samplingrate/2), 15/(samplingrate/2)], btype='bandpass')
    padlen = 3 * (max(len(a), len(b)) - 1)

    all_epochs_list = []
    labels_list = []

    for n_train in range(len(data['train'])):
        t2 = data['train'][n_train]

        # lets us see where target stimulus and where non target stim is
        target_indices = np.where(t2["markers_target"] == 1)[0]
        nontarget_indices = np.where(t2["markers_target"] == 2)[0]

        time = np.arange(epoch_length) / samplingrate * 1000

        raw_data = t2['data'][selected_channels, :]
        
        # Extract all target trials
        for stim in target_indices:
            if stim - baseline_length > 0 and stim + epoch_length <= raw_data.shape[1]:
                baseline = np.mean(raw_data[:, stim-baseline_length:stim+1], axis=1)

                epoch = raw_data[:, stim:stim+epoch_length]

                epoch = epoch - baseline[:, None]

                epoch = detrend(epoch, axis=1, type='linear')

                epoch = filtfilt(b, a, epoch, axis=1, padtype='odd', padlen=padlen)
                
                epoch = epoch.astype(np.float32, copy=False)

                all_epochs_list.append(epoch)
                labels_list.append(1)

        # Extract all nontarget trials
        for stim in nontarget_indices:
            if stim - baseline_length > 0 and stim + epoch_length <= raw_data.shape[1]:
                baseline = np.mean(raw_data[:, stim-baseline_length:stim+1], axis=1)

                epoch = raw_data[:, stim:stim+epoch_length]

                epoch = epoch - baseline[:, None]

                epoch = detrend(epoch, axis=1, type='linear')

                epoch = filtfilt(b, a, epoch, axis=1, padtype='odd', padlen=padlen)

                epoch = epoch.astype(np.float32, copy=False)

                all_epochs_list.append(epoch)
                labels_list.append(0)

    # Compile dataset
    all_epochs = np.stack(all_epochs_list, axis=2)
    labels = np.array(labels_list, dtype=np.int64)


    # Save dataset as NPZ (Python format)
    if save_path is not None:
        savemat(save_path,
            {'all_epochs': all_epochs,
            'labels': labels,
            'time': time})
        print("Preprocessed EEG dataset saved successfully.")

    return all_epochs, labels, time

def preprocess_testing(file_path, use_training = False, save_path = None):
    """
    preprocesses EEG data for use in testing

    Inputs:
    - file_path: path to the .mat file containing the training dataset
    - use_training: use training set instead of testing set for preprocessing. Defaults False
    - save_path: optional path to save the file in .mat format. defaults to None to not save it

    Returns:
    - all_epochs: numpy array containing the preprocessed EEG signal data in shape (channels, time, signal)
    - all_flash_ids: row/column ID corresponding to the epoch at the same index in all_epochs
    """
    data = mat73.loadmat(file_path)

    if use_training:
        t1 = data['train']
    else:
        t1 = data['test']

    print("Raw data shape:", t1[0]['data'].shape)

    samplingrate = float(t1[0]['srate'])

    baseline_samples = round(0.2 * samplingrate)
    epoch_samples = round(0.6 * samplingrate)

    # =========================
    # FILTER DESIGN (MATLAB MATCH)
    # =========================
    b, a = butter(4, [0.5/(samplingrate/2), 15/(samplingrate/2)], btype='bandpass')
    padlen = 3 * (max(len(a), len(b)) - 1)

    selected_channels = np.array([31,32,12,13,19,14,18,16]) - 1

    time = np.arange(epoch_samples) / samplingrate * 1000

    all_epochs = []
    all_flash_ids = []
    for n in range(len(data)):
        t2 = t1[n]

        raw_data = t2['data'][selected_channels, :]

        markers_seq = np.array(t2['markers_seq'])

        flash_indices = np.where((markers_seq >= 1) & (markers_seq <= 12))[0]
        flash_ids = markers_seq[flash_indices]

        # =========================
        # EPOCH LOOP
        # =========================
        for i in range(len(flash_indices)):

            stim = flash_indices[i]

            if stim - baseline_samples > 0 and stim + epoch_samples <= raw_data.shape[1]:

                baseline = np.mean(raw_data[:, stim-baseline_samples:stim+1], axis=1)

                epoch = raw_data[:, stim:stim+epoch_samples]

                epoch = epoch - baseline[:, None]

                epoch = detrend(epoch, axis=1, type='linear')

                epoch = filtfilt(b, a, epoch, axis=1, padtype='odd', padlen=padlen)

                all_epochs.append(epoch)
                all_flash_ids.append(flash_ids[i])

    all_epochs = np.stack(all_epochs, axis=2)
    all_flash_ids = np.array(all_flash_ids)

    # =========================
    # SAVE PYTHON OUTPUT
    # =========================
    if save_path is not None:
        savemat(save_path,
                {'all_epochs': all_epochs,
                'all_flash_ids': all_flash_ids,
                'time': time})

        print('Python dataset saved :)')

    return all_epochs, all_flash_ids


def find_times(file_path):
    data = mat73.loadmat(file_path)
    samplingrate = data['train'][0]['srate']
    epoch_length = int(round(0.6 * samplingrate))

    time = np.arange(epoch_length) / samplingrate * 1000  # milliseconds

    print("Times min:", time.min())
    print("Times max:", time.max())
    print("Times shape:", time.shape)
    return time