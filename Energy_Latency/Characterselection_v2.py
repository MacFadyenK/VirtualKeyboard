import numpy as np  # imported for matrix math
import torch
import scipy.special as sp
import time
import h5py
import os

# The 6 x 6 grid of all character options. 
# To select for only 8 characters, it would be best to reduce the matrix size (ex: 3 x 3).
letters = [
    ["A","B","C","D","E","F"],
    ["G","H","I","J","K","L"],
    ["M","N","O","P","Q","R"],
    ["S","T","U","V","W","X"],
    ["Y","Z","1","2","3","4"],
    ["5","6","7","8","9","_"]
]

# Metrics Constants from your prior code
FLOPS_PER_ELEMENT_CHARSEL = 5 
ENERGY_COEFF = 5.6e-12 

def create_flash_matrix_probs(tensor, y): #(time, sample, num_outputs)
    # For 1D
    flash_matrix = np.zeros(12) # Initialize 12 0's.
    
    flops_select = 0
    
    tensor = tensor.permute(1, 0, 2) # changes to (sample, time, num_outputs)
    tensor = tensor.cpu().numpy() # Convert the tensor to a NumPy array for easier manipulation. This assumes the tensor is on a GPU; if it's already on the CPU, this step can be skipped. 

    logits = tensor.mean(axis=1) # FLOPs
    flops_select += logits.size

    probs = sp.softmax(logits, axis=1) # Apply the softmax function to the logits to convert them into probabilities.
    
    
    
    
    flops_select += 5*logits.size # from google... look into
    
    
    
    
    
    P300_probs = probs[:, 1] # Extract the probabilities for the hit class (P300) for each flash.
    
    # 1D array 
    for index in range(len(P300_probs)):
        if y[index] <= 6: # row flash
            # Add the hit count to the corresponding row in the flash matrix.
            flash_matrix[int(y[index])-1] += np.log(P300_probs[index] + 1e-8) 
        else: # column flash
            # Add the hit count to the corresponding column in the flash matrix. 
            flash_matrix[int(y[index])-1] += np.log(P300_probs[index] + 1e-8) 
            
    flops_select += 3*len(P300_probs)

    return flash_matrix, flops_select  # Return after the loop


# P300 speller cycle character selection function
def p300_speller_cycle_prob_with_metrics(tensor, y): 

    # 2. Latency Measurement: Start high-resolution timer
    t_start = time.perf_counter()

    flash_matrix, flops_select = create_flash_matrix_probs(tensor, y)

    # 1D array
    row_totals = flash_matrix[:6]
    col_totals = flash_matrix[6:]

    # Choose max row and column
    row_idx = np.argmax(row_totals)
    col_idx = np.argmax(col_totals)

    # Using the indices of the selected row and column to retrieve the corresponding letter from the letters matrix.
    predicted_letter = letters[row_idx][col_idx] 

    t_end = time.perf_counter()
    latency = t_end - t_start

    return {
        "predicted_letter": predicted_letter,
        "row_idx": row_idx,
        "col_idx": col_idx,
        "row_totals": row_totals,
        "col_totals": col_totals,
        "flops": flops_select,
        "latency": latency
    }

# ============================================================
# MAIN BLOCK (Modified to handle your v7.3 file structure)
# ============================================================

if __name__ == "__main__":
    file_name = "s17_spikes.mat" 
    
    if not os.path.exists(file_name):
        print(f"File '{file_name}' not found. Check your directory.")
    else:
        try:
            # Using h5py for MATLAB v7.3 files
            with h5py.File(file_name, 'r') as f:
                # Based on your previous error, data is nested in 'test'
                tensor_np = np.array(f["test/tensor"]).T 
                y = np.array(f["test/y"]).squeeze()

            tensor = torch.from_numpy(tensor_np).float()

            # Run logic and get metrics
            results = p300_speller_cycle_prob_with_metrics(tensor, y)

            print(f"===== PROBABILISTIC SELECTION REPORT =====")
            print(f"Predicted Letter: {results['predicted_letter']}")
            print(f"Total FLOPs:      {results['flops']:.3e}")
            print(f"Latency:          {results['latency']*1000:.6f} ms")
            print(f"Energy:           {results['energy']:.6e} Joules")

        except Exception as e:
            print(f"An error occurred: {e}")