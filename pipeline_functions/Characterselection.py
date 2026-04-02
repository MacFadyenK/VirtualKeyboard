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

    #For 1D
    flash_matrix = np.zeros(12) #Initialize 12 0's.

    #For 2D
    #flash_matrix = np.zeros((6, 6)) #Initialize a 6x6 matrix of zeros to represent the flash pattern.
    
    #tensor = tensor.detach().clone() if isinstance(tensor, torch.Tensor) else torch.tensor(tensor) #Include if given dimension issues
    tensor = tensor.permute(1, 0, 2) # changes to (sample, time, num_outputs)
    tensor = tensor.cpu().numpy() # Convert the tensor to a NumPy array for easier manipulation. This assumes the tensor is on a GPU; if it's already on the CPU, this step can be skipped. 

    hits_per_flash = tensor[:, :, 1].sum(axis=1) #Extract the hit counts for each flash from the tensor for the second column P300.(samples)
    #print(f"Hits per flash: {hits_per_flash}") #Print the hit counts for each flash to verify the values.

    non_hits_per_flash = tensor[:, :, 0].sum(axis=1) #Extract the non-hit counts for each flash from the tensor for the first column (non-P300).
    #print(f"Non-hits per flash: {non_hits_per_flash}") #Print the non-hit counts for each flash to verify the values.

    diff_per_flash = hits_per_flash - non_hits_per_flash #Calculate the difference between hits and non-hits for each flash to get a more accurate representation of the signal strength for each flash.
    #print(f"Difference per flash: {diff_per_flash}") #Print the difference between hits and non-hits for each flash to verify the values.

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

#2D array saved
#for index in range(len(hits_per_flash)):
        #print(y[index])
        #if y[index] <= 6: #row flash
            #flash_matrix[y[index]-1, :] += hits_per_flash[index] #Add the hit count to the corresponding row in the flash matrix. Subtract 1 from y[index] to convert from 1-based indexing to 0-based indexing used in Python.
        #else: #column flash
            #flash_matrix[:, y[index]-7] += hits_per_flash[index] #Add the hit count to the corresponding column in the flash matrix. Subtract 7 from y[index] to convert from 1-based indexing to 0-based indexing and account for the first 6 rows.

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

#Commented out for BCI keyboard pipeline. 
#tensor = []
#y= 0

#Printing the letter with the highest score in the row and column.
#predicted_letter, row_idx, col_idx, row_scores, col_scores = p300_speller_cycle(tensor, y) 

#print(f"Row scores: {row_scores}")
#print(f"Column scores: {col_scores}")
#print(f"Selected Row: {row_idx}, Selected Column: {col_idx}")
#print(f"Predicted letter: {predicted_letter}")

def best_of_flash_matrix(tensor, y): #(time, sample, num_outputs)

    #For 1D
    flash_matrix = np.zeros((3, 12)) #Initialize a 3x12 matrix of zeros.

    #For 2D
    #flash_matrix = np.zeros((6, 6)) #Initialize a 6x6 matrix of zeros to represent the flash pattern.
    
    #tensor = tensor.detach().clone() if isinstance(tensor, torch.Tensor) else torch.tensor(tensor) #Include if given dimension issues
    tensor = tensor.permute(1, 0, 2) # changes to (sample, time, num_outputs)
    tensor = tensor.cpu().numpy() # Convert the tensor to a NumPy array for easier manipulation. This assumes the tensor is on a GPU; if it's already on the CPU, this step can be skipped. 

    hits_per_flash = tensor[:, :, 1].sum(axis=1) #Extract the hit counts for each flash from the tensor for the second column P300.(samples)
    #print(f"Hits per flash: {hits_per_flash}") #Print the hit counts for each flash to verify the values.

    non_hits_per_flash = tensor[:, :, 0].sum(axis=1) #Extract the non-hit counts for each flash from the tensor for the first column (non-P300).
    #print(f"Non-hits per flash: {non_hits_per_flash}") #Print the non-hit counts for each flash to verify the values.

    diff_per_flash = hits_per_flash - non_hits_per_flash #Calculate the difference between hits and non-hits for each flash to get a more accurate representation of the signal strength for each flash.
    #print(f"Difference per flash: {diff_per_flash}") #Print the difference between hits and non-hits for each flash to verify the values.

    #Adjusted verison without uses the nested for loops. Can go back if needed. 
    #1D array 
    r = 0
    for index in range(len(hits_per_flash)):
        # print(y[index])
        if y[index] <= 6: #row flash
            flash_matrix[r, y[index]-1] += diff_per_flash[index] #Add the hit count to the corresponding row in the flash matrix. Subtract 1 from y[index] to convert from 1-based indexing to 0-based indexing used in Python.
           # print(f"Updated flash matrix after row flash: {flash_matrix}") #Print the updated flash matrix after processing a row flash to verify the changes.
        else: #column flash
            flash_matrix[r, y[index]-1] += diff_per_flash[index] #Add the hit count to the corresponding column in the flash matrix. Subtract 1 from y[index] to convert from 1-based indexing to 0-based indexing
            #print(f"Updated flash matrix after column flash: {flash_matrix}") #Print the updated flash matrix after processing a column flash to verify the changes.
        if index % 12  == 11:
            r += 1
            #print(f"Flash matrix after cycle {r}: {flash_matrix}") #Print the flash matrix after each cycle of 12 flashes to track the changes over time.
    
    return flash_matrix  # Return after the loop

def p300_speller_cycle_bestof(tensor, y): 

    flash_matrix = best_of_flash_matrix(tensor, y)

    #1D array
    row_totals = flash_matrix[:, :6]
    col_totals = flash_matrix[:, 6:]

    row_sums = flash_matrix[:, :6].sum(axis=0) #Sum the row scores for each cycle to get a total score for each row across all cycles.
    col_sums = flash_matrix[:, 6:].sum(axis=0) #Sum the column scores for each cycle to get a total score for each column across all cycles.

    #print(f"Row sums: {row_sums}")
    #print(f"Column sums: {col_sums}")
    # Sum across repetitions for 2D array
    #row_totals = np.sum(flash_matrix, axis=1) #Set to axis=1 to sum across rows, giving a total score for each row. This will help identify which row has the most hits.
    #col_totals = np.sum(flash_matrix, axis=0) #set to axis=0 to sum across columns, giving a total score for each column. This will help identify which column has the most hits.

    # Choose max row with tie-breaking
    row_idx = np.argmax(row_totals, axis=1) #Find the index of the maximum value in each row to identify the most likely row for each cycle.
    unique_rows, row_counts = np.unique(row_idx, return_counts=True) #Count the occurrences of each row index to determine which row was selected most frequently across cycles.
    
    max_row_freq = np.max(row_counts) #Find the maximum frequency of row selection to identify how many times the most frequently selected row was chosen across cycles.
    ties_row = unique_rows[row_counts == max_row_freq] #Identify any ties in row selection by finding all row indices that have the maximum frequency of selection.
    
    if len(ties_row) > 1: #If there are ties in row selection
        #print(f"Ties in row selection: {ties_row}") #Print the tied row indices to verify the tie-breaking process.
        #print(row_sums[ties_row]) #Print the total scores for the tied rows to verify the values used for tie-breaking.
        best_row_idx = ties_row[np.argmax(row_sums[ties_row])] #Select the row index with the highest total score among the tied rows as the best row.
    else:
        #print(f"No ties in row selection. Unique rows: {unique_rows}, Row counts: {row_counts}") #Print the unique row indices and their counts to verify the selection process when there are no ties.
        best_row_idx = unique_rows[np.argmax(row_counts)] #If there are no ties, select the row index that was selected most frequently as the best row.
    
    #Choose max column with tie-breaking
    col_idx = np.argmax(col_totals, axis=1) #Find the index of the maximum value in each column to identify the most likely column for each cycle.
    unique_cols, col_counts = np.unique(col_idx, return_counts=True) #Count the occurrences of each column index to determine which column was selected most frequently across cycles.

    max_col_freq = np.max(col_counts) #Find the maximum frequency of column selection to identify how many times the most frequently selected column was chosen across cycles.
    ties_col = unique_cols[col_counts == max_col_freq] #Identify any ties in column selection by finding all column indices that have the maximum frequency of selection.
    
    if len(ties_col) > 1: #If there are ties in column selection
        #print(f"Ties in column selection: {ties_col}") #Print the tied column indices to verify the tie-breaking process.
        #print(col_sums[ties_col]) #Print the total scores for the tied columns to verify the values used for tie-breaking.
        best_col_idx = ties_col[np.argmax(col_sums[ties_col])] #Select the column index with the highest total score among the tied columns as the best column.
    else:
        #print(f"No ties in column selection. Unique columns: {unique_cols}, Column counts: {col_counts}") #Print the unique column indices and their counts to verify the selection process when there are no ties.
        best_col_idx = unique_cols[np.argmax(col_counts)] #If there are no ties, select the column index that was selected most frequently as the best column.

    predicted_letter = letters[best_row_idx][best_col_idx] #Using the indices of the selected row and column to retrieve the corresponding letter from the letters matrix.

    #print(f"Row Index: {best_row_idx}, Column Index: {best_col_idx}")
    return predicted_letter, best_row_idx, best_col_idx, row_totals, col_totals

def p300_speller_cycle_rank(tensor, y): 

    flash_matrix = best_of_flash_matrix(tensor, y)

    #1D array
    row_totals = flash_matrix[:, :6]
    col_totals = flash_matrix[:, 6:]

    row_ranks = np.argsort(row_totals, axis=1)[:, ::-1] #Rank the row totals for each cycle in descending order to identify the most likely rows for each cycle.
    col_ranks = np.argsort(col_totals, axis=1)[:, ::-1] #Rank the column totals for each cycle in descending order to identify the most likely columns for each cycle.

    #print(f"Row ranks: {row_ranks}") #Print the row ranks to verify the ranking of rows for each cycle.
    #print(f"Column ranks: {col_ranks}") #Print the column ranks to verify the ranking of columns for each cycle.

    ranker = np.zeros((3, 12)) #Initialize a 3x12 matrix of zeros to store the ranks of each row and column for each cycle.
    for i in range(3):
        for j in range(6):
            ranker[i, row_ranks[i, j]] += j+1 # Assign rank to each row based on its position in the sorted order
            ranker[i, col_ranks[i, j]+6] += j+1#Assign a rank to each column based on its position in the sorted order, adding 6 to account for the first 6 rows in the flash matrix.
    
    #print(f"Ranker matrix before averaging: {ranker}") #Print the ranker matrix before averaging to verify the assigned ranks.

    ranker = np.mean(ranker, axis=0) #Average the ranks across cycles to get an overall rank for each row and column.

    #print(f"Ranker matrix after averaging: {ranker}") #Print the ranker matrix after averaging to verify the overall ranks for each row and column.
    
    ranker_rows = ranker[:6] #Extract the ranks for the rows from the ranker matrix.
    ranker_cols = ranker[6:] #Extract the ranks for the columns from the ranker matrix.

    # Sum across repetitions for 2D array
    #row_totals = np.sum(flash_matrix, axis=1) #Set to axis=1 to sum across rows, giving a total score for each row. This will help identify which row has the most hits.
    #col_totals = np.sum(flash_matrix, axis=0) #set to axis=0 to sum across columns, giving a total score for each column. This will help identify which column has the most hits.

    # Choose max row and column
    row_idx = np.argmin(ranker_rows) #Find the index of the minimum averaged rank in the row ranks to identify the most likely row
    col_idx = np.argmin(ranker_cols) #Find the index of the minimum averaged rank in the column ranks to identify the most likely column

    predicted_letter = letters[row_idx][col_idx]


    return predicted_letter, row_idx, col_idx, ranker_rows, ranker_cols