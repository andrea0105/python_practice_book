import numpy as np

np.random.seed(10)

A = np.random.randint(0, 9, 12).reshape(4, 3)
b = np.random.randint(0, 9, 3).reshape(3, 1)

C = np.dot(A, b)
C_ = np.array([A[:,[j]] * b[j] for j in range(A.shape[1])])
C__ = C_.sum(axis=0)
print(C)
print(C__)