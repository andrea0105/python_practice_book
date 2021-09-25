import numpy as np

np.random.seed(2)

m, n = 4, 3
A = np.random.randint(0, 10, m*n).reshape(m, n)
b = np.random.randint(0, 10, n+1)
C = b[:, np.newaxis]
D = A * C

print(b)
print(C)
print(D)