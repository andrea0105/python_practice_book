import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1)

def f(X):
    print(X[0], X[1])
    return X[0]**2 + X[1]**2 + 2

x_0 = np.array([0.7, 0.7])
d = np.array([-2, -2])

def f_alpha(a):
    return f(x_0.reshape(2, 1) + a * d.reshape(2, 1))


alphas = np.linspace(0, 1, 30)
ax.plot(alphas, f_alpha(alphas), 'k')
ax.set_xlabel(r'$\alpha$', fontsize=20)
ax.set_ylabel(r'$f(\alpha)$', fontsize=20)
plt.savefig('steepest.png')

