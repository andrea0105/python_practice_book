import numpy as np

A = np.arange(1, 7).reshape(3, 2)
print(A)

B = A.transpose(1, 0)
print(B)