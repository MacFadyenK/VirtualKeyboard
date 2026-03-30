import numpy as np
import mat73
from scipy.io import savemat
from scipy.signal import butter, filtfilt, detrend

# =========================
# USER SETTINGS (CHANGE HERE ONLY)
# =========================
mode = "all"        # "all" or "one", preprocess full dataset or just one character at a time
target_char = "B"   # used to select target charater, only if mode = "one"

data_path = r'C:\Users\chloe\Desktop\s17.mat'
save_path_all = r'C:\Users\chloe\Desktop\s17_Full.mat'
save_path_one = r'C:\Users\chloe\Desktop\s17_OneChar.mat'


# =========================
# FUNCTIONS
# =========================
def preprocess_all_characters(t2):
    samplingrate = float(t2['srate'])
    baseline_samples = round(0.2 * samplingrate)
    epoch_samples = round(0.6 * samplingrate)

    b, a = butter(4, [0.5 / (samplingrate / 2), 15 / (samplingrate / 2)], btype='bandpass')

    selected_channels = np.array([31, 32, 12, 13, 19, 14, 18, 16]) - 1
    time = np.arange(epoch_samples) / samplingrate * 1000

    raw_data = t2['data'][selected_channels, :]
    markers_seq = np.array(t2['markers_seq'])

    flash_indices = np.where((markers_seq >= 1) & (markers_seq <= 12))[0]
    flash_ids = markers_seq[flash_indices]

    all_epochs = []
    all_flash_ids = []

    for i in range(len(flash_indices)):
        stim = flash_indices[i]

        if stim - baseline_samples > 0 and stim + epoch_samples <= raw_data.shape[1]:
            baseline = np.mean(raw_data[:, stim - baseline_samples:stim + 1], axis=1)

            epoch = raw_data[:, stim:stim + epoch_samples]
            epoch = epoch - baseline[:, None]
            epoch = detrend(epoch, axis=1, type='linear')

            padlen = 3 * (max(len(a), len(b)) - 1)
            epoch = filtfilt(b, a, epoch, axis=1, padtype='odd', padlen=padlen)

            all_epochs.append(epoch)
            all_flash_ids.append(flash_ids[i])

    all_epochs = np.stack(all_epochs, axis=2)
    all_flash_ids = np.array(all_flash_ids)

    correct_chars = list(t2['text_to_spell'])

    flashes_per_char = len(all_flash_ids) // len(correct_chars)
    all_char_labels = []

    for i in range(len(all_flash_ids)):
        char_idx = min(i // flashes_per_char, len(correct_chars) - 1)
        all_char_labels.append(correct_chars[char_idx])

    all_char_labels = np.array(all_char_labels)

    return all_epochs, all_flash_ids, time, all_char_labels, correct_chars


def preprocess_one_character(t2, target_char):
    samplingrate = float(t2['srate'])
    baseline_samples = round(0.2 * samplingrate)
    epoch_samples = round(0.6 * samplingrate)

    b, a = butter(4, [0.5 / (samplingrate / 2), 15 / (samplingrate / 2)], btype='bandpass')

    selected_channels = np.array([31, 32, 12, 13, 19, 14, 18, 16]) - 1
    time = np.arange(epoch_samples) / samplingrate * 1000

    raw_data = t2['data'][selected_channels, :]
    markers_seq = np.array(t2['markers_seq'])

    flash_indices = np.where((markers_seq >= 1) & (markers_seq <= 12))[0]
    flash_ids = markers_seq[flash_indices]

    correct_chars = list(t2['text_to_spell'])

    if target_char not in correct_chars:
        raise ValueError(f"'{target_char}' not in {''.join(correct_chars)}")

    flashes_per_char = len(flash_indices) // len(correct_chars)

    char_positions = [i for i, c in enumerate(correct_chars) if c == target_char]

    char_epochs = []
    char_flash_ids = []
    char_labels = []

    for char_pos in char_positions:
        start_idx = char_pos * flashes_per_char
        end_idx = (char_pos + 1) * flashes_per_char

        for i in range(start_idx, end_idx):
            stim = flash_indices[i]

            if stim - baseline_samples > 0 and stim + epoch_samples <= raw_data.shape[1]:
                baseline = np.mean(raw_data[:, stim - baseline_samples:stim + 1], axis=1)

                epoch = raw_data[:, stim:stim + epoch_samples]
                epoch = epoch - baseline[:, None]
                epoch = detrend(epoch, axis=1, type='linear')

                padlen = 3 * (max(len(a), len(b)) - 1)
                epoch = filtfilt(b, a, epoch, axis=1, padtype='odd', padlen=padlen)

                char_epochs.append(epoch)
                char_flash_ids.append(flash_ids[i])
                char_labels.append(target_char)

    char_epochs = np.stack(char_epochs, axis=2)
    char_flash_ids = np.array(char_flash_ids)
    char_labels = np.array(char_labels)

    return char_epochs, char_flash_ids, time, char_labels


# =========================
# MAIN
# =========================
data = mat73.loadmat(data_path)
t2 = data['train'][0]

print("Raw data shape:", t2['data'].shape)

if mode == "all":
    all_epochs, all_flash_ids, time, all_char_labels, correct_chars = preprocess_all_characters(t2)

    print("Correct chars:", correct_chars)
    print("Epochs shape:", all_epochs.shape)

    savemat(save_path_all, {
        'all_epochs': all_epochs,
        'all_flash_ids': all_flash_ids,
        'all_char_labels': all_char_labels,
        'time': time
    })

    print("Saved full dataset ✅")

elif mode == "one":
    char_epochs, char_flash_ids, time, char_labels = preprocess_one_character(t2, target_char)

    print("Epochs shape:", char_epochs.shape)

    savemat(save_path_one, {
        'char_epochs': char_epochs,
        'char_flash_ids': char_flash_ids,
        'char_labels': char_labels,
        'time': time
    })

    print(f"Saved dataset for '{target_char}' ✅")

