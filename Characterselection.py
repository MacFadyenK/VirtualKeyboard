letters = ["A" "E" "H" "N" "O" "R" "S" "T"] 
import numpy as np #imported for matrix math
n = 8 #eight characters 
prior = np.ones(n)/n
#creates array of eight 1's and divides by 8
likelihood = np.array[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
#even distrubtion for now, could change in the future