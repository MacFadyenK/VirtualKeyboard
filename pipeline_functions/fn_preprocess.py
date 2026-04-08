import numpy as np
from scipy.signal import butter, filtfilt, detrend
from scipy.io import savemat
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


    # Save dataset as .mat file
    if save_path is not None:
        savemat(save_path,
            {'all_epochs': all_epochs,
            'labels': labels,
            'time': time})
        print("Preprocessed EEG dataset saved successfully.")

    return all_epochs, labels, time

def preprocess_testing(file_path, use_training = False, set = [0, 4], save_path = None):
    """
    preprocesses EEG data for use in testing

    Inputs:
    - file_path: path to the .mat file containing the training dataset
    - use_training: use training set instead of testing set for preprocessing. Defaults False
    - set: list containing the start and end indices for the subset of data to process
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
    all_char_labels = ''
    for n in range(set[0], min(set[1], len(t1))):
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

        correct_chars = t2['text_to_spell']
        all_char_labels += correct_chars

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

    return all_epochs, all_flash_ids, all_char_labels


def preprocess_one_character(t2):
    """
    preprocesses EEG data for one character for character selection

    Inputs:
    - t2: a dictionary containing the data for one character, including the raw EEG data, sampling rate, marker sequence, and the character itself

    Returns:
    - char_epochs: numpy array containing the preprocessed EEG signal data in shape (channels, time, epochs)
    - char_flash_ids: labels corresponding to the flash ID of each epoch in char_epochs (1-12)
    - time: the time each epoch takes up
    - correct_char: the character that was intended to be selected
    """
    # getting baseline and epoch lengths in samples
    samplingrate = float(t2['srate'])
    baseline_samples = round(0.2 * samplingrate)
    epoch_samples = round(0.6 * samplingrate)

    # create filter and padding length for bandpass filter
    b, a = butter(4, [0.5 / (samplingrate / 2), 15 / (samplingrate / 2)], btype='bandpass')
    padlen = 3 * (max(len(a), len(b)) - 1)

    # select channels for use and create time vector for epoch
    selected_channels = np.array([31, 32, 12, 13, 19, 14, 18, 16]) - 1
    time = np.arange(epoch_samples) / samplingrate * 1000

    raw_data = t2['data'][selected_channels, :]

    # get marker sequence and flash indices
    markers_seq = np.array(t2['markers_seq'])

    flash_indices = np.where((markers_seq >= 1) & (markers_seq <= 12))[0]
    flash_ids = markers_seq[flash_indices]

    # get the character
    correct_char = t2['text_to_spell']

    char_epochs = []
    # extract epochs for the target character
    for i in range(len(flash_indices)):
        stim = flash_indices[i]

        if stim - baseline_samples > 0 and stim + epoch_samples <= raw_data.shape[1]:
            baseline = np.mean(raw_data[:, stim - baseline_samples:stim + 1], axis=1)

            epoch = raw_data[:, stim:stim + epoch_samples]
            epoch = epoch - baseline[:, None]
            epoch = detrend(epoch, axis=1, type='linear')

            epoch = filtfilt(b, a, epoch, axis=1, padtype='odd', padlen=padlen)

            char_epochs.append(epoch)

    # compile character epochs and flash ids into arrays
    char_epochs = np.stack(char_epochs, axis=2)
    char_flash_ids = np.array(flash_ids)

    return char_epochs, char_flash_ids, time, correct_char


def load_subject_character_data(file_path, letter):
    """
    Loads the raw EEG data for a specific character indicated. If multiple instances of the character are found, one is randomly selected for loading.
    
    Inputs:
    - file_path: path to the .mat file containing the dataset
    - letter: the character to load data for (e.g., 'A', 'B', etc.)
    
    Returns:
    - a dictionary containing the raw EEG data, sampling rate, marker sequence, and the character itself
    """
    data = mat73.loadmat(file_path)

    t_test = data['test']
    
    # searching through data for the desired character
    found_in = [] # list of test sets where the target letter is found
    for n_test in range(len(t_test)):
        t2 = t_test[n_test]
        if letter in t2['text_to_spell']:
            # print(f"Found {letter} in test set {n_test}")
            found_in.append(n_test)
    if not found_in:
        raise ValueError(f"Letter {letter} not found in any test set.")
    
    # randomly select one of the test sets where the letter is found
    selected_test = np.random.choice(found_in)
    # print(f"Randomly selected test set {selected_test} for letter {letter}")
    
    # get data from that test set
    t2 = t_test[selected_test]
    sr = t2['srate']
    all_chars = t2['text_to_spell']

    # get all positions of the target character in this set
    char_positions = [i for i, c in enumerate(all_chars) if c == letter]

    # randomly choose an instance of the target character if it appears multiple times
    position_index = np.random.choice(char_positions)
    # print(f"Randomly selected position {position_index} for letter {letter} in test set {selected_test}")
    
    markers_seq = np.array(t2['markers_seq'])

    flash_indices = np.where((markers_seq >= 1) & (markers_seq <= 12))[0]

    flashes_per_char = len(flash_indices) // len(all_chars)
    
    # indices where the character starts and ends in the flash sequence
    start_flash_idx = position_index * flashes_per_char
    end_flash_idx = (position_index + 1) * flashes_per_char

    # indices for character selected including prestimulus for baseline 250ms before first flash up to 1000ms after the last flash of the character
    start_index = flash_indices[start_flash_idx] - round(.250*sr)
    end_index = flash_indices[end_flash_idx-1] + round(1*sr)

    character_data = t2['data'][:, start_index:end_index]
    character_marker_seq = markers_seq[start_index:end_index]
    
    return {
        'data': character_data,
        'srate': sr,
        'markers_seq': character_marker_seq,
        'text_to_spell': letter,
    }