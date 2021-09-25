import numpy as np

np.random.seed(10)

n, m = 5, 2

X = np.random.rand(n, m)
w = np.random.rand(m)
y = np.random.rand(n)

g = np.zeros_like(w)

'''for j in range(m):
    for i in range(n):
        g[j] += (np.dot(X[i,:], w) - y[i]) * X[i,j]'''

g = np.dot(X.T, np.dot(X, w) - y)

print(g)