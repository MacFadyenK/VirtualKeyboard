import numpy as np
import mat73
from scipy.io import savemat, loadmat
from scipy.signal import butter, filtfilt, detrend

# =========================
# LOAD MATLAB FILE (RAW DATA)
# =========================
data = mat73.loadmat('/Users/mikepasamba/Desktop/MATLAB_Preprocessing/s17.mat')



t2 = data['train'][0]

print("Raw data shape:", t2['data'].shape)

samplingrate = float(t2['srate'])

baseline_samples = round(0.2 * samplingrate)
epoch_samples = round(0.6 * samplingrate)

# =========================
# FILTER DESIGN (MATLAB MATCH)
# =========================
b, a = butter(4, [0.5/(samplingrate/2), 15/(samplingrate/2)], btype='bandpass')

selected_channels = np.array([31,32,12,13,19,14,18,16]) - 1

time = np.arange(epoch_samples) / samplingrate * 1000

raw_data = t2['data'][selected_channels, :]

markers_seq = np.array(t2['markers_seq'])

flash_indices = np.where((markers_seq >= 1) & (markers_seq <= 12))[0]
flash_ids = markers_seq[flash_indices]

all_epochs = []
all_flash_ids = []

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

        padlen = 3 * (max(len(a), len(b)) - 1)
        epoch = filtfilt(b, a, epoch, axis=1, padtype='odd', padlen=padlen)

        all_epochs.append(epoch)
        all_flash_ids.append(flash_ids[i])

all_epochs = np.stack(all_epochs, axis=2)
all_flash_ids = np.array(all_flash_ids)

# =========================
# SAVE PYTHON OUTPUT
# =========================
savemat('/Users/mikepasamba/Desktop/MATLAB_Preprocessing/S17_FlashEpochs_Preprocessing_PYTHON.mat',
        {'all_epochs': all_epochs,
         'all_flash_ids': all_flash_ids,
         'time': time})

print('Python dataset saved :)')

# =========================
# LOAD MATLAB OUTPUT FOR COMPARISON
# =========================
m = loadmat('/Users/mikepasamba/Desktop/MATLAB_Preprocessing/S17_FlashEpochs_Preprocessing_MATLAB.mat')

print("MATLAB shape:", m['all_epochs'].shape)
print("Python shape:", all_epochs.shape)

diff = np.abs(m['all_epochs'] - all_epochs)

print("MAX DIFFERENCE:", np.max(diff))
print("MEAN DIFFERENCE:", np.mean(diff))

# =========================
# GET CORRECT CHARACTERS IN ORDER
# =========================
correct_chars = list(t2['text_to_spell'])
# output of correct chars = ['B', 'R', 'A', 'I', 'N']

flashes_per_char = len(all_flash_ids) // len(correct_chars)
all_char_labels = []
for i in range(len(all_flash_ids)):
    char_idx = min(i // flashes_per_char, len(correct_chars) - 1)
    all_char_labels.append(correct_chars[char_idx])
all_char_labels = np.array(all_char_labels)
# output of all_char_labels =
    # epochs 0–179 → 'B'
    # epochs 180–359 → 'R'
    # epochs 360–539 → 'A'
    # epochs 540–719 → 'I'
    # epochs 720–899 → 'N'
