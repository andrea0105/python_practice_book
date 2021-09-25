import numpy as np

A = np.matrix([[2, 1], [-6 ,3]])
B = np.matrix([[3], [-27]])

C = A.I * B
print(B)