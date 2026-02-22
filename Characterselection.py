import numpy as np  # imported for matrix math


#Simulated P300 likelihood function
def simulated_p300_likelihood(flash_index, target_letter, letters, matrix):
    #flash index: which row of the matrix is currently flashing
    #target_letter: the letter we want to predict
    #letters: list of all letters in the speller
    #matrix: the flash pattern matrix

    target_index = letters.index(target_letter)
    #Where to find the target letter in the flash matrix. Ex: A = 0, E = 1, etc.
    flash_letters = matrix[flash_index]
    #Selects the current flashing row. 1 = flash, 0 = no flash

    if flash_letters[target_index] == 1:
    #Means the target letter is flashing, so we simulate a strong P300 response.
        return np.random.uniform(0.7, 0.9)  # strong response
    else:
        return np.random.uniform(0.0, 0.2)  # weak/no response


#P300 speller cycle character selection function
def p300_speller_cycle(target_letter, repetition=10):
    # Repetitions is number of flash cycles. Can be changed to increase accuracy (10–15 recommended)

    letters = ["A", "E", "H", "N", "O", "R", "S", "T"]
    #The current 8 letter.
    n = len(letters)
    #Stores the eight letters.

    probabilities = np.ones(n) / n
    # uniform probabilities for each letter at the start (1/8 for each).

    # flash matrix (8 flashes × 8 letters). Each row is one flash. Each column is one letter.
    matrix = np.array([
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [1, 0, 0, 1, 1, 0, 0, 1],
        [0, 1, 0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0, 1]
    ])

    # flash cycles
    for _ in range(repetition): #repeat the flashing process for the specified number of repetitions.
        for flash_index in range(matrix.shape[0]): #for each flash in the matrix (8 flashes total)

            likelihood = simulated_p300_likelihood(
                flash_index, target_letter, letters, matrix
            ) #Simuated the response of the likelihood. 

            # Bayesian-style update. So that flashed letters have a higher probability.
            probabilities *= (1 + likelihood * matrix[flash_index])

            # normalize
            probabilities /= probabilities.sum()

    predicted_letter = letters[np.argmax(probabilities)] #chooses the letter with the highest probability as the predicted letter.
    return predicted_letter, probabilities

#Testing the letters 
letters = ["A", "E", "H", "N", "O", "R", "S", "T"]

print("Testing all target letters:\n")

for target in letters:
    predicted, probs = p300_speller_cycle(target)
    print(f"Target: {target} | Predicted: {predicted}")