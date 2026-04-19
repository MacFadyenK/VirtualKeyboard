# ============================================================
# ANN PIPELINE: PREPROCESSING -> FEATURES -> INFERENCE
# ============================================================
import numpy as np
import torch
import mat73
import time
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend

# Custom imports
from ANNModule import createANN, run_ann_with_metrics
from FeatureExtraction import extractFeatures

# --- Configuration ---
POWER_WATTS = 20.0
ENERGY_COEFF = 5.6e-12
FILE_PATH = "s17.mat"
TARGET_LETTER = "M"
WEIGHT_PATH = "ann_model_weights.pth"

# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================

def load_subject_character_data(file_path, letter):
    data = mat73.loadmat(file_path)
    t_test = data['test']
    found_in = [n for n, t2 in enumerate(t_test) if letter in t2['text_to_spell']]
    if not found_in: raise ValueError(f"Letter {letter} not found.")
    selected_test = np.random.choice(found_in)
    t2 = t_test[selected_test]
    sr = t2['srate']
    all_chars = t2['text_to_spell']
    char_pos = [i for i, c in enumerate(all_chars) if c == letter]
    pos_idx = np.random.choice(char_pos)
    markers = np.array(t2['markers_seq'])
    flash_idx = np.where((markers >= 1) & (markers <= 12))[0]
    flashes_per_char = len(flash_idx) // len(all_chars)
    s_idx = flash_idx[pos_idx * flashes_per_char] - round(.250*sr)
    e_idx = flash_idx[(pos_idx + 1) * flashes_per_char - 1] + round(1*sr)
    return {'data': t2['data'][:, s_idx:e_idx], 'srate': sr, 'markers_seq': markers[s_idx:e_idx], 'text_to_spell': letter}

def preprocess_one_character(t2):
    t_start = time.time()
    f_count = 0
    sr = float(t2['srate'])
    b_smp, e_smp = round(0.2 * sr), round(0.6 * sr)
    b, a = butter(4, [0.5/(sr/2), 15/(sr/2)], btype='bandpass')
    # Use standard 8 P300 channels
    raw = t2['data'][[30, 31, 11, 12, 18, 13, 17, 15], :] 
    markers = np.array(t2['markers_seq'])
    f_idx = np.where((markers >= 1) & (markers <= 12))[0]
    f_ids = markers[f_idx]
    epochs, ids = [], []
    for stim, fid in zip(f_idx, f_ids):
        if stim - b_smp > 0 and stim + e_smp <= raw.shape[1]:
            seg_b = raw[:, stim-b_smp:stim+1]
            base = np.mean(seg_b, axis=1)
            ep = raw[:, stim:stim+e_smp] - base[:, None]
            ep = detrend(ep, axis=1)
            ep = filtfilt(b, a, ep, axis=1)
            epochs.append(ep)
            ids.append(fid)
            f_count += seg_b.size + ep.size * 26 
    return np.stack(epochs, axis=2), np.array(ids), time.time() - t_start, f_count

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # 1. Initialize Model
    ann = createANN(136, [128, 64], num_outputs=2)
    checkpoint = torch.load(WEIGHT_PATH, weights_only=True)
    ann.load_state_dict(checkpoint['model_state_dict'])
    ann.eval()

    # Metrics Trackers
    stats = {'pre_l': [], 'feat_l': [], 'ann_l': []}
    ops = {'pre': 0, 'feat': 0, 'ann': 0}

    for i in range(5):
        print(f"Iteration {i+1}/5...")
        
        # Step A: Load Raw Character Data
        t_data = load_subject_character_data(FILE_PATH, TARGET_LETTER)
        
        # Step B: Preprocessing (Bandpass, Detrend, Baseline)
        ep, f_ids, l_pre, f_pre = preprocess_one_character(t_data)
        stats['pre_l'].append(l_pre * 1000)
        ops['pre'] = f_pre

        # Step C: Feature Extraction (Decimation & Normalization)
        X, y, l_feat, f_feat = extractFeatures(ep, f_ids, t_min=200, t_max=400, norm_type='std', factor=18)
        
        # Shape Guard: Ensure (Flashes, Channels, Time) matches model expectations
        if X.shape[2] != 17:
            X = X[:,:,:17] if X.shape[2] > 17 else np.pad(X, ((0,0),(0,0),(0, 17-X.shape[2])))
        
        stats['feat_l'].append(l_feat * 1000)
        ops['feat'] = f_feat

        # Step D: ANN Inference
        # Note: run_ann_with_metrics handles its own internal timing
        (out, _, l_ann, _, _, _, f_ann) = run_ann_with_metrics(ann, X, power_watts=POWER_WATTS)
        
        stats['ann_l'].append(l_ann * 1000)
        ops['ann'] = f_ann

    # 3. Aggregate Statistics
    l_avgs = [np.mean(stats[k]) for k in ['pre_l', 'feat_l', 'ann_l']]
    l_stds = [np.std(stats[k], ddof=1) for k in ['pre_l', 'feat_l', 'ann_l']]
    e_vals = [ops[k] * ENERGY_COEFF * 1e6 for k in ['pre', 'feat', 'ann']]

    labels = ["Start", "Preprocessing", "Feature\nExtraction", "ANN\nInference"]
    l_ind, l_err = [0] + l_avgs, [0] + l_stds
    l_cum = np.cumsum(l_ind)
    e_ind, e_cum = [0] + e_vals, np.cumsum([0] + e_vals)

    # 4. Final Plots
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Latency Plot
    ax1.bar(labels, l_ind, yerr=l_err, color='#bbd0ff', alpha=0.8, capsize=8, ecolor='red', label="Stage Latency")
    ax1.plot(labels, l_cum, marker='o', color='#4361ee', linewidth=3, label="Cumulative Latency")
    ax1.set_title(f"ANN Execution Latency (Total: {l_cum[-1]:.2f} ms)", fontsize=14, pad=15)
    ax1.set_ylabel("Latency (ms)")
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.legend()
    for i, c in enumerate(l_cum):
        ax1.annotate(f"{c:.1f}", (labels[i], c), xytext=(0, 12), textcoords="offset points", ha='center', color='#4361ee')

    # Energy Plot
    ax2.bar(labels, e_ind, color='#b7e4c7', alpha=0.8, label="Stage Energy")
    ax2.plot(labels, e_cum, marker='s', color='#2d6a4f', linewidth=3, label="Cumulative Energy")
    ax2.set_title(f"ANN Execution Energy (Total: {e_cum[-1]:.2f} \u03bcJ)", fontsize=14, pad=15)
    ax2.set_ylabel("Energy (\u03bcJ)")
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.legend()
    for i, c in enumerate(e_cum):
        ax2.annotate(f"{c:.2f}", (labels[i], c), xytext=(0, 12), textcoords="offset points", ha='center', color='#2d6a4f')

    plt.tight_layout()
    plt.show()

    print(f"\nAnalysis Complete.")
    print(f"Total Latency: {l_cum[-1]:.2f} ms")
    print(f"Total Energy:  {e_cum[-1]:.2f} uJ")