import numpy as np #imported for matrix math
def simulated_p300_likelihood(flash_index, target_letter, letters, matrix):
    target_index = letters.index(target_letter)
    flash_letters = matrix[flash_index]
    if flash_letters[target_index] == 1:
        return np.random.uniform(0.7, 0.9)  # strong response
    else:
        return np.random.uniform(0.0, 0.2)  # weak/no response
def p300_speller_cycle(p300_likelihood, repetition= 10):
    #Defines functintion P(P300 | flash). repetitions can be changed to increase accuracy. 10-15 recommended. 
    letters = ["A","E","H","N","O","R","S","T"]
    n = len(letters) #eight characters based on index position 0-7. 
    probabilities = np.ones(n)/n
    #creates array of eight 1's and divides by 8
    matrix = np.array([[0, 1, 0, 1, 0, 1, 0, 1], 
                       [1, 0, 1, 0, 1, 0, 1, 0], 
                       [0, 0, 1, 1, 0, 0, 1, 1], 
                       [1, 1, 0, 0, 1, 1, 0, 0], 
                       [0, 1, 1, 0, 0, 1, 1, 0], 
                       [1, 0, 0, 1, 1, 0, 0, 1], 
                       [0, 1, 0, 1, 1, 0, 1 ,0], 
                       [1 ,0 ,1 ,0 ,0 ,1 ,0 ,1]])
    #matrix of which characters are flashed in each trial. 1 is flashed, 0 is not. Placeholder for now. 8 flashes, can be adjusted. 
    for _ in range(repetition):
    #For flash cycles. 
        for flash_index in range(matrix.shape[0]):
            #To run through each flash cycle. Matrix.shape[0] = number of rows = number of flashes (8)
            likelihood = p300_likelihood(flash_index)
            probabilities *= (1 + likelihood * matrix[flash_index])
            probabilities /= probabilities.sum()
            predicted_letter = letters[np.argmax(probabilities)]
            #To maximize the array value of probability. 
    return predicted_letter, probabilities 
#Still working on printing 