import numpy as np  # imported for matrix math
import torch
import scipy.special as sp

letters = [
        ["A","B","C","D","E","F"],
        ["G","H","I","J","K","L"],
        ["M","N","O","P","Q","R"],
        ["S","T","U","V","W","X"],
        ["Y","Z","1","2","3","4"],
        ["5","6","7","8","9","_"]] 
    #The 6 x 6 grid of all character options. To select for only 8 characters, it would be best to reduce the matrix size (ex: 3 x 3).

def create_flash_matrix_probs(tensor, y): #(time, sample, num_outputs)
    #For 1D
    flash_matrix = np.zeros(12) #Initialize 12 0's.
    
    tensor = tensor.permute(1, 0, 2) # changes to (sample, time, num_outputs)
    tensor = tensor.cpu().numpy() # Convert the tensor to a NumPy array for easier manipulation. This assumes the tensor is on a GPU; if it's already on the CPU, this step can be skipped. 

    logits = tensor.mean(axis=1) #mean of the spikes across the time dimension to get a total score for each flash.

    probs = sp.softmax(logits, axis=1) #Apply the softmax function to the logits to convert them into probabilities.
    P300_probs = probs[:, 1] #Extract the probabilities for the hit class (P300) for each flash.
    #print(f"P300 probabilities per flash: {P300_probs}") #Print the P300 probabilities for each flash to verify the values.
    #1D array 
    for index in range(len(P300_probs)):
        #print(y[index])
        if y[index] <= 6: #row flash
            flash_matrix[y[index]-1] += np.log(P300_probs[index] + 1e-8) #Add the hit count to the corresponding row in the flash matrix.
            #print(f"Updated flash matrix after row flash: {flash_matrix}")
        else: #column flash
            flash_matrix[y[index]-1] += np.log(P300_probs[index] + 1e-8) #Add the hit count to the corresponding column in the flash matrix. 
            #print(f"Updated flash matrix after column flash: {flash_matrix}") 

    return flash_matrix  # Return after the loop


#P300 speller cycle character selection function
def p300_speller_cycle_prob(tensor, y): 

    flash_matrix = create_flash_matrix_probs(tensor, y)

    #1D array
    row_totals = flash_matrix[:6]
    col_totals = flash_matrix[6:]

    # Choose max row and column
    row_idx = np.argmax(row_totals)
    col_idx = np.argmax(col_totals)

    predicted_letter = letters[row_idx][col_idx] #Using the indices of the selected row and column to retrieve the corresponding letter from the letters matrix.

    return predicted_letter, row_idx, col_idx, row_totals, col_totals
