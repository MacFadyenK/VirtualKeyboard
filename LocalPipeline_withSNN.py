# ============================================================
# CHARACTER SELECTION PIPELINE USING SNN
# ============================================================

import numpy as np
import torch
import mat73
import time
from scipy.signal import butter, filtfilt, detrend
from scipy.special import softmax
from SNNModule import createSNN, run_snn_with_metrics, count_snn_sops
from FeatureExtraction import extractFeatures
from DeltaEncoding import delta_encode
from Characterselection_v2 import p300_speller_cycle_prob_with_metrics, create_flash_matrix_probs
import matplotlib.pyplot as plt


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
    print("preprocessing → feature extraction → delta encoding → SNN → Character Selection...\n")

    file_path = "s17.mat"
    target_letter = "M"

    snn = createSNN(8, [128, 64], betas=[0.95, 0.95, 0.95], thresholds=[1, 1, 1])
    weights_path = "12864_th09_f3_stdf1_weights.pth"
    weights = torch.load(weights_path, weights_only=True)
    snn.load_state_dict(weights)
    snn.eval()
   
    latencies_pre = []
    latencies_feat = []
    latencies_enc = []
    latencies_snn = []
    latencies_select = []
    for i in range(5):
        
        # ---------------------------------------------------------
        # 1. Load raw EEG for the selected character
        # ---------------------------------------------------------
        t2 = load_subject_character_data(file_path, target_letter)
    
        # ---------------------------------------------------------
        # 2. Preprocess → (8, 307, 180) + flash IDs
        # ---------------------------------------------------------
        char_epochs_x, char_flash_ids_y, latency_pre, flops_pre = preprocess_one_character(t2)
        latencies_pre.append(latency_pre*1000)
        
        # ---------------------------------------------------------
        # 3. Feature Extraction
        # ---------------------------------------------------------
        X_feat, y_feat, latency_feat, flops_feat = extractFeatures(
            dataset=char_epochs_x,
            y=char_flash_ids_y,
            factor=3,
            t_min=200,
            t_max=400,
            norm_type='std'
        )
        latencies_feat.append(latency_feat*1000)
    
        # ---------------------------------------------------------
        # 4. Delta Encoding
        # ---------------------------------------------------------
        x_encoded, latency_enc, flops_enc = delta_encode(X_feat)
        latencies_enc.append(latency_enc*1000)
        
        # ---------------------------------------------------------
        # 5. SNN
        # ---------------------------------------------------------
        energy_coeff = 5.6e-12 # This is the estimated energy in Joules per operation
        X_spikes = torch.from_numpy(x_encoded).float()
        y = y_feat.astype(int)
        (spkout, lats, latency_snn, 
         epoch_ops, total_snn_ops, 
         snn_energy, total_sops) = run_snn_with_metrics(snn, X_spikes, power_watts=POWER_WATTS)        
        latencies_snn.append(latency_snn*1000)
    
        # ---------------------------------------------------------     
        # 6. Character Selection
        # ---------------------------------------------------------
        metrics_results = p300_speller_cycle_prob_with_metrics(spkout, y_feat)
        selectedchar = metrics_results['predicted_letter']
        latency_select = metrics_results['latency']
        flops_select = metrics_results['flops']
        latencies_select.append(latency_select*1000)
        
    
    latencies_pre = np.array(latencies_pre)
    latencies_feat = np.array(latencies_feat)
    latencies_enc = np.array(latencies_enc)
    latencies_snn = np.array(latencies_snn)
    latencies_select = np.array(latencies_select)
    
    latency_avg_pre = np.mean(latencies_pre)
    latency_avg_feat = np.mean(latencies_feat)
    latency_avg_enc = np.mean(latencies_enc)
    latency_avg_snn = np.mean(latencies_snn)
    latency_avg_select = np.mean(latencies_select)
    
    latency_std_pre = np.std(latencies_pre, ddof=1)
    latency_std_feat = np.std(latencies_feat, ddof=1)
    latency_std_enc = np.std(latencies_enc, ddof=1)
    latency_std_snn = np.std(latencies_snn, ddof=1)
    latency_std_select = np.std(latencies_select, ddof=1)

    print("Preprocessing Latency:", round(latency_avg_pre), 'ms')
    print("Preprocessing FLOPs:", flops_pre, "\n")

    
    print("Feature Extraction Latency:", round(latency_avg_feat), "ms")
    print("Feature Extraction FLOPs:", flops_feat, "\n")
    
    print("Delta Encoding Latency:", round(latency_avg_enc), "ms")
    print("Delta Encoding FLOPs:", flops_enc, "\n")

    
    print("SNN Latency:", round(latency_avg_snn), "ms")
    print("SNN SOPs:", int(total_sops))
    print(f"SNN Energy: {total_sops*energy_coeff*1000000:.1f} uJ", "\n")
    
    
    print("Character Selection Latency:", round(latency_avg_select), "ms")
    print("Character Selection FLOPs:", flops_select, "\n")
   
    print("FINAL RESULTS")
    print("-----------------------------------------")
    print("Target Character:", target_letter)
    print("Predicted Character:", selectedchar, "\n")
    print("Total Latency:", round((latency_avg_pre + latency_avg_feat + latency_avg_enc + latency_avg_snn + latency_avg_select)), "ms")
    print(f"Total Operations: {(flops_pre + flops_feat + flops_enc + total_sops + flops_select):.0f} ")
    print("Estimated Energy:", round((flops_pre + flops_feat + flops_enc + total_sops + flops_select)*energy_coeff*1000000), "uJ")

    # ---------------------------------------------------------      
    # 7. Latency: Individual Bars (with Std Dev) + Cumulative Line
    # ---------------------------------------------------------
    labels = [
        "Start", 
        "Pre-\nprocessing", 
        "Feature\nExtraction", 
        "Delta\nEncoding", 
        "SNN\nProcessing", 
        "Character\nSelection"
    ]
    
    # Individual averages and standard deviations
    ind_durations = [0, latency_avg_pre, latency_avg_feat, latency_avg_enc, latency_avg_snn, latency_avg_select]
    individual_stds = [0, latency_std_pre, latency_std_feat, latency_std_enc, latency_std_snn, latency_std_select]
    
    # Cumulative running total
    cum_durations = np.cumsum(ind_durations)

    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.figure(figsize=(14, 8))
    
    # 1. Plot Individual Bars with Error Bars (Standard Deviation)
    # yerr=individual_stds adds the vertical lines you requested
    bars = plt.bar(labels, ind_durations, yerr=individual_stds, 
                   color='#a0c4ff', alpha=0.6, capsize=8, ecolor='red', 
                   label="Stage Latency (Mean ± SD)")
    
    # 2. Plot Cumulative Line
    plt.plot(labels, cum_durations, marker='o', color='#1f77b4', 
             linewidth=3, markersize=10, label="Total Latency (Cumulative)")

    # Formatting and Styling
    plt.title("Latency Analysis: Stage-wise Variability vs. System Total", fontsize=16, fontweight='bold', pad=30)
    plt.ylabel("Latency (ms)", fontsize=13, fontweight='bold', labelpad=15)
    plt.xlabel("Pipeline Operation", fontsize=13, fontweight='bold', labelpad=20)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', prop={'weight':'bold'})

    # Annotations
    for i in range(len(labels)):
        # Label Individual Mean above the bars (accounting for the error bar height)
        if ind_durations[i] > 0:
            offset = individual_stds[i] + (max(ind_durations) * 0.05)
            plt.text(i, ind_durations[i] + offset, f"{ind_durations[i]:.1f}ms", 
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='#4361ee')
        
        # Label Cumulative Total below the points
        plt.annotate(f"Total: {cum_durations[i]:.1f}ms", (labels[i], cum_durations[i]), 
                     xytext=(0, -25), textcoords="offset points", 
                     ha='center', va='top', fontweight='bold', color='black')

    plt.ylim(bottom=-20, top=max(cum_durations) * 1.2)
    plt.tight_layout()
    plt.show()
    
    
    # ---------------------------------------------------------      
    # 8. Energy: Cumulative Line + Individual Bars
    # ---------------------------------------------------------
    # Individual energy per stage (uJ)
    ind_energies = [
        0, 
        flops_pre * energy_coeff * 1e6,
        flops_feat * energy_coeff * 1e6,
        flops_enc * energy_coeff * 1e6,
        total_sops * energy_coeff * 1e6,
        flops_select * energy_coeff * 1e6
    ]
    cum_energies = np.cumsum(ind_energies)

    plt.figure(figsize=(14, 8))
    
    # Plotting Individual Bars
    plt.bar(labels, ind_energies, color='#b7e4c7', alpha=0.6, label="Stage Energy (Individual)")
    
    # Plotting Cumulative Line
    plt.plot(labels, cum_energies, marker='o', color='#2ca02c', linewidth=3, markersize=10, label="Total Energy (Cumulative)")

    # Formatting
    plt.title("Energy Analysis: Individual vs. Cumulative", fontsize=16, fontweight='bold', pad=25)
    plt.ylabel("Energy (\u03bcJ)", fontsize=13, fontweight='bold', labelpad=15)
    plt.xlabel("Pipeline Operation", fontsize=13, fontweight='bold', labelpad=20)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', prop={'weight':'bold'})

    # Annotations
    for i in range(len(labels)):
        # Individual value above bar
        if ind_energies[i] > 0:
            plt.text(i, ind_energies[i] + (max(ind_energies)*0.02), f"{ind_energies[i]:.2f}\u03bcJ", 
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1b4332')
        # Cumulative value below point
        plt.annotate(f"Total: {cum_energies[i]:.2f}\u03bcJ", (labels[i], cum_energies[i]), 
                     xytext=(0, -25), textcoords="offset points", ha='center', va='top', fontweight='bold')

    plt.ylim(bottom=-max(cum_energies)*0.1)
    plt.tight_layout()
    plt.show()