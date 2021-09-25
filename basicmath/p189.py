import numpy as np

np.random.seed(12)

A = np.random.randint(0, 9, 4).reshape(2, 2)
B = np.random.randint(0, 9, 4).reshape(2, 2) + 1

print(A)
print(B)
print(A+1)
print(np.dot(A, B))