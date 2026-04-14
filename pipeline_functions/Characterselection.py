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

def create_flash_matrix(tensor, y): #(time, sample, num_outputs)

    #For 1D
    flash_matrix = np.zeros(12) #Initialize 12 0's.

    #For 2D
    #flash_matrix = np.zeros((6, 6)) #Initialize a 6x6 matrix of zeros to represent the flash pattern.
    
    #tensor = tensor.detach().clone() if isinstance(tensor, torch.Tensor) else torch.tensor(tensor) #Include if given dimension issues
    tensor = tensor.permute(1, 0, 2) # changes to (sample, time, num_outputs)
    tensor = tensor.cpu().numpy() # Convert the tensor to a NumPy array for easier manipulation. This assumes the tensor is on a GPU; if it's already on the CPU, this step can be skipped. 

    hits_per_flash = tensor[:, :, 1].mean(axis=1) #Extract the hit counts for each flash from the tensor for the second column P300.(samples)
    #print(f"Hits per flash: {hits_per_flash}") #Print the hit counts for each flash to verify the values.

    non_hits_per_flash = tensor[:, :, 0].mean(axis=1) #Extract the non-hit counts for each flash from the tensor for the first column (non-P300).
    #print(f"Non-hits per flash: {non_hits_per_flash}") #Print the non-hit counts for each flash to verify the values.

    diff_per_flash = hits_per_flash - non_hits_per_flash #Calculate the difference between hits and non-hits for each flash to get a more accurate representation of the signal strength for each flash.
    #print(f"Difference per flash: {diff_per_flash}") #Print the difference between hits and non-hits for each flash to verify the values.

    #hits_per_flash = (hits_per_flash - np.mean(hits_per_flash)) / (np.std(hits_per_flash) + 1e-8) #Normalize the hit counts for each flash to have a mean of 0 and a standard deviation of 1, which can help improve the performance of the character selection algorithm by reducing the impact of outliers and scaling the values appropriately.
    #Adjusted verison without uses the nested for loops. Can go back if needed. 
    #1D array 
    for index in range(len(hits_per_flash)):
        #print(y[index])
        if y[index] <= 6: #row flash
            flash_matrix[y[index]-1] += diff_per_flash[index] #Add the hit count to the corresponding row in the flash matrix. Subtract 1 from y[index] to convert from 1-based indexing to 0-based indexing used in Python.
            #print(f"Updated flash matrix after row flash: {flash_matrix}") #Print the updated flash matrix after processing a row flash to verify the changes.
        else: #column flash
            flash_matrix[y[index]-1] += diff_per_flash[index] #Add the hit count to the corresponding column in the flash matrix. Subtract 1 from y[index] to convert from 1-based indexing to 0-based indexing and account for the first 6 rows.
            #print(f"Updated flash matrix after column flash: {flash_matrix}") #Print the updated flash matrix after processing a column flash to verify the changes.

    return flash_matrix  # Return after the loop

#P300 speller cycle character selection function
def p300_speller_cycle(tensor, y): 

    flash_matrix = create_flash_matrix(tensor, y)

    #1D array
    row_totals = flash_matrix[:6]
    col_totals = flash_matrix[6:]

    # Sum across repetitions for 2D array
    #row_totals = np.sum(flash_matrix, axis=1) #Set to axis=1 to sum across rows, giving a total score for each row. This will help identify which row has the most hits.
    #col_totals = np.sum(flash_matrix, axis=0) #set to axis=0 to sum across columns, giving a total score for each column. This will help identify which column has the most hits.

    # Choose max row and column
    row_idx = np.argmax(row_totals)
    col_idx = np.argmax(col_totals)

    predicted_letter = letters[row_idx][col_idx] #Using the indices of the selected row and column to retrieve the corresponding letter from the letters matrix.

    return predicted_letter, row_idx, col_idx, row_totals, col_totals

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
            flash_matrix[y[index]-1] += np.log(P300_probs[index]) #Add the hit count to the corresponding row in the flash matrix.
            #print(f"Updated flash matrix after row flash: {flash_matrix}")
        else: #column flash
            flash_matrix[y[index]-1] += np.log(P300_probs[index]) #Add the hit count to the corresponding column in the flash matrix. 
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
