# ============================================================
# CHARACTER SELECTION PIPELINE USING SNN
# ============================================================
import numpy as np
import torch
import mat73
import time
from scipy.signal import butter, filtfilt, detrend
from scipy.special import softmax
from ANNModule import createANN, run_ann_with_metrics, count_ann_flops
from FeatureExtraction import extractFeatures
from DeltaEncoding import delta_encode
from CharacterSelection import p300_speller_cycle_prob, create_flash_matrix_probs


POWER_WATTS = 20.0
K_AVG = 5   # flashes to average per flash ID (must match training)


# ============================================================
# LETTER MATRIX
# ============================================================

letters = np.array([
    ['A','B','C','D','E','F'],
    ['G','H','I','J','K','L'],
    ['M','N','O','P','Q','R'],
    ['S','T','U','V','W','X'],
    ['Y','Z','1','2','3','4'],
    ['5','6','7','8','9','0']
])


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def decimation_by_avg(data, factor):
    n_trial, n_ch, n_frame = data.shape
    decimated_frame = int(np.floor(n_frame / factor))
    decimated = np.zeros((n_trial, n_ch, decimated_frame))

    for i in range(n_trial):
        for j in range(decimated_frame):
            decimated[i, :, j] = np.mean(
                data[i, :, j*factor:(j+1)*factor],
                axis=1
            )
    return decimated


def extract_features_option1(X):
    # X: (flashes, 8, ~307) → (flashes, 8, 34)

    # 1) Downsample
    X = decimation_by_avg(X, factor=5)

    # 2) Window 105–440 ms within 600 ms epoch
    t_step = 600 / X.shape[2]
    step_min = round(105 / t_step)
    step_max = round(440 / t_step)
    X = X[:, :, step_min:step_max]

    # 3) Normalize per channel
    eps = 1e-8
    ch_min = X.min(axis=(0, 2), keepdims=True)
    ch_max = X.max(axis=(0, 2), keepdims=True)
    X_norm = (X - ch_min) / (ch_max - ch_min + eps)

    return X_norm   # (flashes, 8, 34)


# ============================================================
# LOAD CHARACTER
# ============================================================

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


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_one_character(t2):
    time_pre = time.time()
    
    # Initialize FLOP counter for this function
    flops_pre = 0
    
    samplingrate = float(t2['srate'])
    baseline_samples = round(0.2 * samplingrate)
    epoch_samples = round(0.6 * samplingrate)

    b, a = butter(4, [0.5/(samplingrate/2), 15/(samplingrate/2)], btype='bandpass')
    padlen = 3 * (max(len(a), len(b)) - 1)

    selected_channels = np.array([31, 32, 12, 13, 19, 14, 18, 16]) - 1
    raw_data = t2['data'][selected_channels, :]

    markers_seq = np.array(t2['markers_seq'])
    flash_indices = np.where((markers_seq >= 1) & (markers_seq <= 12))[0]
    flash_ids_all = markers_seq[flash_indices]

    char_epochs = []
    char_flash_ids = []

    for stim, fid in zip(flash_indices, flash_ids_all):
        if stim - baseline_samples > 0 and stim + epoch_samples <= raw_data.shape[1]:
            # 1. Baseline Mean Calculation
            # FLOPs: Number of elements in the baseline window
            segment_baseline = raw_data[:, stim-baseline_samples:stim+1]
            baseline = np.mean(segment_baseline, axis=1)
            flops_pre += segment_baseline.size 

            # 2. Baseline Subtraction
            # FLOPs: 1 subtraction per element in the epoch
            epoch = raw_data[:, stim:stim+epoch_samples]
            epoch = epoch - baseline[:, None]
            flops_pre += epoch.size

            # 3. Detrending
            # FLOPs: Estimated 5 per element (slope, intercept, trend removal)
            epoch = detrend(epoch, axis=1)
            flops_pre += epoch.size * 5

            # 4. Filtering (filtfilt)
            # FLOPs: Estimated 20 per element (forward & backward pass)
            epoch = filtfilt(b, a, epoch, axis=1, padtype='odd', padlen=padlen)
            flops_pre += epoch.size * 20

            char_epochs.append(epoch)
            char_flash_ids.append(fid)

    char_epochs = np.stack(char_epochs, axis=2)  # (8, 307, flashes)
    char_flash_ids = np.array(char_flash_ids)    # (flashes,)

    valid = (char_flash_ids >= 1) & (char_flash_ids <= 12)
    char_epochs = char_epochs[:, :, valid]
    char_flash_ids = char_flash_ids[valid].astype(int)
    
    latency_pre = time.time() - time_pre
    
    return char_epochs, char_flash_ids, latency_pre, flops_pre


# ============================================================
# FLASH-WISE AVERAGING (MATCH TRAINING)
# ============================================================

def average_by_flash_id(X_feat, flash_ids, k=5):
    """
    X_feat: (flashes, 8, 34)
    flash_ids: (flashes,)
    returns:
      X_avg: (n_avg, 8, 34)
      y_avg: (n_avg,)
    """
    from collections import defaultdict
    buffers = defaultdict(list)
    X_avg, y_avg = [], []

    for xi, yi in zip(X_feat, flash_ids):
        buffers[int(yi)].append(xi)
        if len(buffers[int(yi)]) == k:
            X_avg.append(np.mean(buffers[int(yi)], axis=0))
            y_avg.append(int(yi))
            buffers[int(yi)].clear()

    if len(X_avg) == 0:
        raise ValueError("No averaged flashes created — check K_AVG or repetitions.")

    return np.stack(X_avg), np.array(y_avg, dtype=int)


# ============================================================
# MAIN PIPELINE WITH METRICS
# ============================================================

if __name__ == "__main__":
    print("")
    print("preprocessing → feature extraction → ANN...\n")

    file_path = "s17.mat"
    target_letter = "M"

    ann = createANN(136, [128, 64], num_outputs=2)
    weights_path = "ann_model_weights.pth"
    # print("Weights Path:", weights_path,"\n")
    checkpoint = torch.load(weights_path, weights_only=True)
    # Access the actual weights stored inside the dictionary
    ann.load_state_dict(checkpoint['model_state_dict'])
    ann.eval()
    
    # ---------------------------------------------------------
    # 1. Load raw EEG for the selected character
    # ---------------------------------------------------------
    t2 = load_subject_character_data(file_path, target_letter)


    # ---------------------------------------------------------
    # 2. Preprocess → (8, 307, 180) + flash IDs
    # ---------------------------------------------------------
    char_epochs_x, char_flash_ids_y, latency_pre, flops_pre = preprocess_one_character(t2)
    # print("Preprocessed epochs:", char_epochs_x.shape)
    # print("Flash IDs:", char_flash_ids_y.shape)
    # print("Preprocessing Latency:", round(latency_pre*1000), 'ms')
    # print("Preprocessing FLOPs:", flops_pre, "\n")


    # ---------------------------------------------------------
    # 3. Feature Extraction
    # ---------------------------------------------------------
    # --- Update your Feature Extraction call ---
    X_feat, y_feat, latency_feat, flops_feat = extractFeatures(
    char_epochs_x, 
    char_flash_ids_y, 
    t_min=200, 
    t_max=400, 
    norm_type='std', 
    factor=18  # This was likely set too high (e.g., 60), resulting in 40 features
)

    # 2. Add a 'Force Shape' check to prevent the crash
    if X_feat.shape[2] != 17:
    # If it's too long, trim it; if too short, pad it with zeros
        if X_feat.shape[2] > 17:
            X_feat = X_feat[:, :, :17]
        else:
            padding = torch.zeros((X_feat.shape[0], X_feat.shape[1], 17 - X_feat.shape[2]))
            X_feat = torch.cat([torch.from_numpy(X_feat), padding], dim=2).numpy()

    # print("Feature Extraction Latency:", round(latency_feat*1000), "ms")
    # print("Feature Extraction FLOPs:", flops_feat, "\n")

        
    # ---------------------------------------------------------
    # 4. ANN
    # ---------------------------------------------------------
    energy_coeff = 5.6e-12 # This is the estimated energy in Joules per operation
    y = y_feat.astype(int)
    
    (spkout, lats, latency_ann, 
     epoch_ops, total_ann_ops, 
     nn_energy, total_ann_flops) = run_ann_with_metrics(ann, X_feat, power_watts=POWER_WATTS)
 

    print(f"ANN Latency: {latency_ann * 1000:.1f} ms")
    print(f"ANN Total FLOPs: {int(total_ann_flops):}")
    print(f"ANN Energy: {total_ann_flops*energy_coeff*1000000:.1f} uJ", "\n")


    
    