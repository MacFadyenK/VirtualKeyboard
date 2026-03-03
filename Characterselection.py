import numpy as np  # imported for matrix math
#Inputs here list of probablities 1-12. Link to second input in code. Order matrix. Argmax of first fix then second. 

letters = [
        ["A","B","C","D","E","F"],
        ["G","H","I","J","K","L"],
        ["M","N","O","P","Q","R"],
        ["S","T","U","V","W","X"],
        ["Y","Z","1","2","3","4"],
        ["5","6","7","8","9","_"]] 
    #The 6 x 6 grid of all character options. To select for only 8 characters, it would be best to reduce the matrix size (ex: 3 x 3).

def create_flash_matrix(tensor):

    flash_matrix = np.zeros((6, 6)) #Initialize a 6x6 matrix of zeros to represent the flash pattern.

    tensor = [] #3D array based on the input data from SNNval. A 15x2 for each row, col that is repeated.
    
    for index,rowcol in tensor: #For the 12 flashes
        is_col = index < 6 #First 6 rows are col flashes, next 6 are row flashes.
        is_row = not is_col #The remaining 6 rows are row flashes. 7-12.
        hits = 0
        for spikeobj in rowcol: #15
            for specrowcol in spikeobj:#15
                hits += specrowcol[1] #summing second column of tensor to get total hits for each flash. 
        if is_col:
            column_index = index % 6
            flash_matrix[:, column_index] += hits #summing the hits for each column flash and adding to the corresponding column in the flash matrix.
        else:
            row_index = index % 6
            flash_matrix[row_index, :] += hits #summing the hits for each row flash and adding to the corresponding row in the flash matrix.
    return flash_matrix

#P300 speller cycle character selection function
def p300_speller_cycle(tensor, repetition=15): # Repetitions is number of flash cycles. Can be changed to increase accuracy (10–15 recommended)

    flash_matrix = create_flash_matrix(tensor)

    # Sum across repetitions
    row_totals = np.sum(flash_matrix, axis=1) #Set to axis=1 to sum across rows, giving a total score for each row. This will help identify which row has the most hits.
    col_totals = np.sum(flash_matrix, axis=0) #set to axis=0 to sum across columns, giving a total score for each column. This will help identify which column has the most hits.

    # Choose max row and column
    row_idx = np.argmax(row_totals)
    col_idx = np.argmax(col_totals)

    predicted_letter = letters[row_idx][col_idx] #Using the indices of the selected row and column to retrieve the corresponding letter from the letters matrix.

    return predicted_letter, row_idx, col_idx, row_totals, col_totals

#Printing the letter with the highest score in the row and column.
predicted_letter, row_idx, col_idx, row_scores, col_scores = p300_speller_cycle(tensor, repetition=15) 

print(f"Row scores: {row_scores}")
print(f"Column scores: {col_scores}")
print(f"Selected Row: {row_idx}, Selected Column: {col_idx}")
print(f"Predicted letter: {predicted_letter}")