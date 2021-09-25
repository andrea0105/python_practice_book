import numpy as np

A = np.random.randint(0, 10, 12).reshape(3, 4)
x = np.random.randint(0, 10, 3)

m = x.shape[0]
y = []
temp = 0
for i in range(A.shape[0]):
    for j in range(m):
        temp += A[j, i] * x[j]
    
    y.append(temp)
    temp = 0

print(A)
print(x)
print(y)