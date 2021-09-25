import numpy as np

A = np.matrix([[2, 4], [4, 2]])
X = np.matrix([[0.7071, -0.7071], [0.7071, 0.7071]])

D = X.I * A * X

print(D)
