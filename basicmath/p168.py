import numpy as np

A = np.arange(1, 7).reshape(2, 3)

C = A.transpose(1, 0)
print(C)

D = A.transpose(1, 0)
print(D)

E = A.T
print(E)