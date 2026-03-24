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

def create_flash_matrix(tensor): #(time, sample, num_outputs)

    flash_matrix = np.zeros((6, 6)) #Initialize a 6x6 matrix of zeros to represent the flash pattern.
    
    tensor = tensor.permute(1, 0, 2) # changes to (sample, time, num_outputs)

    tensor = tensor.cpu().numpy() # Convert the tensor to a NumPy array for easier manipulation. This assumes the tensor is on a GPU; if it's already on the CPU, this step can be skipped. 

    #Adjusted verison without uses the nested for loops. Can go back if needed. 
    hits_per_flash = tensor[:, :, 1].sum(axis=1) #Extract the hit counts for each flash from the tensor for the second column P300.(samples)
    for index in range(len(hits_per_flash)):
        if (index % 12) < 6:  # row flashes
            flash_matrix[index, :] += hits_per_flash[index]
        else:          # column flashes
            flash_matrix[:, index - 6] += hits_per_flash[index]
    return flash_matrix

#P300 speller cycle character selection function
def p300_speller_cycle(tensor): 

    flash_matrix = create_flash_matrix(tensor)

    # Sum across repetitions
    row_totals = np.sum(flash_matrix, axis=1) #Set to axis=1 to sum across rows, giving a total score for each row. This will help identify which row has the most hits.
    col_totals = np.sum(flash_matrix, axis=0) #set to axis=0 to sum across columns, giving a total score for each column. This will help identify which column has the most hits.

    # Choose max row and column
    row_idx = np.argmax(row_totals)
    col_idx = np.argmax(col_totals)

    predicted_letter = letters[row_idx][col_idx] #Using the indices of the selected row and column to retrieve the corresponding letter from the letters matrix.

    return predicted_letter, row_idx, col_idx, row_totals, col_totals

tensor = []

#Printing the letter with the highest score in the row and column.
predicted_letter, row_idx, col_idx, row_scores, col_scores = p300_speller_cycle(tensor) 

print(f"Row scores: {row_scores}")
print(f"Column scores: {col_scores}")
print(f"Selected Row: {row_idx}, Selected Column: {col_idx}")
print(f"Predicted letter: {predicted_letter}")