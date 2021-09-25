import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[2, 1], [1, 2], [1, 1]])

z = np.dot(B, A)
print(z)

C = np.matrix([[1, 2, 3], [4, 5, 6]])
D = np.matrix([[2, 1], [1, 2], [1, 1]])

f = D * C
print(f)