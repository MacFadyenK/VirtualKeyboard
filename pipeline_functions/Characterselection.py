import numpy as np  # imported for matrix math
import torch

letters = [
        ["A","B","C","D","E","F"],
        ["G","H","I","J","K","L"],
        ["M","N","O","P","Q","R"],
        ["S","T","U","V","W","X"],
        ["Y","Z","1","2","3","4"],
        ["5","6","7","8","9","_"]] 
    #The 6 x 6 grid of all character options. To select for only 8 characters, it would be best to reduce the matrix size (ex: 3 x 3).

def create_flash_matrix(tensor, y): #(time, sample, num_outputs)

    flash_matrix = np.zeros((6, 6)) #Initialize a 6x6 matrix of zeros to represent the flash pattern.
    
    tensor = tensor.detach().clone() if isinstance(tensor, torch.Tensor) else torch.tensor(tensor)
    tensor = tensor.permute(1, 0, 2) # changes to (sample, time, num_outputs)
    tensor = tensor.cpu().numpy() # Convert the tensor to a NumPy array for easier manipulation. This assumes the tensor is on a GPU; if it's already on the CPU, this step can be skipped. 

    hits_per_flash = tensor[:, :, 1].sum(axis=1) #Extract the hit counts for each flash from the tensor for the second column P300.(samples)

    # y = np.array(y).flatten()  # Ensure y is a 1D array for easier indexing. This flattens the array in case it's not already one-dimensional.
    # y = np.clip(y, 0, 11).astype(int)  # safely clip to 0-11

    #Adjusted verison without uses the nested for loops. Can go back if needed. 
    for idx in range(len(hits_per_flash)):
        y_val = y[idx]
        if y_val <=6:       # row flash
            flash_matrix[y_val - 1, :] += hits_per_flash[idx]
        elif 6 < y_val <= 12:    # column flash
            flash_matrix[:, y_val - 7] += hits_per_flash[idx]
        # ignore invalid y_val

    return flash_matrix  # Return after the loop

#P300 speller cycle character selection function
def p300_speller_cycle(tensor, y): 

    flash_matrix = create_flash_matrix(tensor, y)

    # Sum across repetitions
    row_totals = np.sum(flash_matrix, axis=1) #Set to axis=1 to sum across rows, giving a total score for each row. This will help identify which row has the most hits.
    col_totals = np.sum(flash_matrix, axis=0) #set to axis=0 to sum across columns, giving a total score for each column. This will help identify which column has the most hits.

    # Choose max row and column
    row_idx = np.argmax(row_totals)
    col_idx = np.argmax(col_totals)

    predicted_letter = letters[row_idx][col_idx] #Using the indices of the selected row and column to retrieve the corresponding letter from the letters matrix.

    return predicted_letter, row_idx, col_idx, row_totals, col_totals


#tensor = []
#y= 0

#Printing the letter with the highest score in the row and column.
#predicted_letter, row_idx, col_idx, row_scores, col_scores = p300_speller_cycle(tensor, y) 

#print(f"Row scores: {row_scores}")
#print(f"Column scores: {col_scores}")
#print(f"Selected Row: {row_idx}, Selected Column: {col_idx}")
#print(f"Predicted letter: {predicted_letter}")