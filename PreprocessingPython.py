import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

data = scipy.io.loadmat(r'C:\Users\MikeK\Downloads\s17_v72.mat', squeeze_me=True, struct_as_record=False)

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

# Extract 600 ms window after stimulus
one_trial = raw_data[:, firstargetstimulus:firstargetstimulus + epoch_length]

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

# Filter single trial
one_trial_filtered = filtfilt(b, a, one_trial, axis=1)

# Plot raw vs filtered
plt.figure()

plt.plot(time, one_trial[pz_index, :], 'k', label='Raw')
plt.plot(time, one_trial_filtered[pz_index, :], 'r', label='Filtered')

plt.legend()
plt.title('Single Trial - Raw vs Bandpass Filtered - 0.5-15Hz')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')

plt.show()

# Access flash order
t24 = data['train'][0]

flash_indices = np.where(t24.markers_seq > 0)[0]
flash_order = t24.markers_seq[flash_indices]

print(flash_order[:30])
