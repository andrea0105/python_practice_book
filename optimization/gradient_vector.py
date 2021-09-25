import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# f(x, y, z) = 0
x = y = np.arange(-1.0, 1.0, 0.05)
X, Y = np.meshgrid(x, y)
Z = -(1/3) * (X**2) * (np.exp(Y))

# 곡선 C에 대한 변수 t와 x(t), y(t), z(t)
t = np.linspace(-0.5, 0.5, 100)
xt = np.array(2 * t)
yt =  np.array(t)
zt = -(1/3) * (xt**2) * (np.exp(yt))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, rstride=3, cstride=3, \
    cmap=plt.cm.gray, edgecolor='gray', zorder=1)

ax.plot3D(xt, yt, '--', lw=2, color='k', zorder=10)

ax.set_zlim(-1, 1)
ax.set_axis_off()
ax.view_init(35, -15)

plt.savefig('gradient_vector.png')