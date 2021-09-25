import numpy as np
from scipy import optimize

def f1(X):
    return (X[0] + 2*X[1] - 7)**2 + (2*X[0] + X[1] - 5)**2

def df1(X):
    dx0 = 2*(X[0] + 2*X[1] - 7) + 4*(2*X[0] + X[1] - 5)
    dx1 = 4*(X[0] + 2*X[1] - 7) + 2*(2*X[0] + X[1] - 5)
    return np.array([dx0, dx1])

def f2(X):
    return 50 * (X[1] - X[0]**2)**2 + (2 - X[0])**2

def df2(X):
    dx0 = -200*X[0]*(X[1]-X[0]**2)-2*(2-X[0])
    dx1 = 100*(X[1]-X[0]**2)
    return np.array([dx0, dx1])

x = np.array([-1, 2])
result_scipy = optimize.fmin_cg(f2, x)

print(result_scipy)
