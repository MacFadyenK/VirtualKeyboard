import numpy as np
from scipy.signal import butter, filtfilt
import h5py

# Open MATLAB file
file = h5py.File(r'C:\Users\MikeK\Downloads\s17.mat', 'r')

# MATLAB structs are stored as references
train_ref = file['train'][0][0]
train = file[train_ref]

# Now access fields
markers_target = np.array(train['markers_target']).squeeze()
raw_data = np.array(train['data'])
raw_data = raw_data.T
samplingrate = int(np.array(train['srate']).squeeze())

# Find target and non-target indices
target_indices = np.where(markers_target == 1)[0]
nontarget_indices = np.where(markers_target == 2)[0]

# Define epoch parameters
epoch_length = int(round(0.6 * samplingrate))

# Bandpass filter (0.5–15 Hz)
b, a = butter(4, [0.5/(samplingrate/2), 15/(samplingrate/2)], btype='bandpass')

# Example electrode
pz_index = 13

# Time vector (ms)
time = np.arange(epoch_length) / samplingrate * 1000

# Create epoch containers
target_epochs = np.zeros((32, epoch_length, len(target_indices)))
nontarget_epochs = np.zeros((32, epoch_length, len(nontarget_indices)))

# Extract target epochs
for i, stim in enumerate(target_indices):
    target_epochs[:, :, i] = raw_data[:, stim:stim + epoch_length]

# Extract non-target epochs
for i, stim in enumerate(nontarget_indices):
    nontarget_epochs[:, :, i] = raw_data[:, stim:stim + epoch_length]

# Filter epochs
filtered_target_epochs = np.zeros_like(target_epochs)
filtered_nontarget_epochs = np.zeros_like(nontarget_epochs)

for i in range(len(target_indices)):
    filtered_target_epochs[:, :, i] = filtfilt(b, a, target_epochs[:, :, i].T).T

for i in range(len(nontarget_indices)):
    filtered_nontarget_epochs[:, :, i] = filtfilt(b, a, nontarget_epochs[:, :, i].T).T

# Average across trials
average_target = np.mean(filtered_target_epochs, axis=2)
average_nontarget = np.mean(filtered_nontarget_epochs, axis=2)

# Compile dataset
numberof_target = filtered_target_epochs.shape[2]
numberof_nontarget = filtered_nontarget_epochs.shape[2]

labels = np.concatenate((np.ones(numberof_target), np.zeros(numberof_nontarget)))

all_epochs = np.concatenate((filtered_target_epochs, filtered_nontarget_epochs), axis=2)

# Save dataset as NPZ (Python format)
np.savez(
    r'C:\Users\MikeK\Downloads\DatasetMatfiles\S17_Preprocessed_Epoch.npz',
    filtered_target_epochs=filtered_target_epochs,
    filtered_nontarget_epochs=filtered_nontarget_epochs,
    all_epochs=all_epochs,
    labels=labels,
    average_target=average_target,
    average_nontarget=average_nontarget,
    samplingrate=samplingrate,
    time=time,
    pz_index=pz_index
)

print("Preprocessed EEG dataset saved successfully.")
