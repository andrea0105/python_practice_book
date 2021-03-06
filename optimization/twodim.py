import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성
np.random.seed(seed=1)
X_min = 4
X_max = 30
X_n = 16
X = 5 + 25 * np.random.rand(X_n) # 나이
Prm_c = [170, 108, 0.2] # Parameters for number generation
T = Prm_c[0] - Prm_c[1] * np.exp(-Prm_c[2] * X) + 4 * np.random.randn(X_n) # 키

#  리스트 5-1-(12)
#  2차원 데이터 생성 ---------------------
X0 = X
X0_min = 5
X0_max = 30
np.random.seed(seed=1)
X1 = 23 * (T/100)**2 + 2 * np.random.randn(X_n) # 몸무게
X1_min = 40
X1_max = 75

# 2차원 데이터의 표시
def show_data2(ax, x0, x1, t):
    for i in range(len(x0)):
        ax.plot([x0[i], x0[i]], [x1[i], x1[i]], [120, t[i]], color='gray')
        ax.plot(x0, x1, t, 'o', color='cornflowerblue', markeredgecolor='black', markersize=6, \
            markeredgewidth=0.5)
        ax.view_init(elev=25, azim=-75)

# 면의 표시
def show_plane(ax, w):
    px0 = np.linspace(X0_min, X0_max, 5)
    px1 = np.linspace(X1_min, X1_max, 5)
    px0, px1 = np.meshgrid(px0, px1)
    y = w[0] * px0 + w[1] * px1 + w[2]
    ax.plot_surface(px0, px1, y, rstride=1, cstride=1, alpha=0.3, \
        color='blue', edgecolor='black')
    
# 면의 Mean Squared Error
def mse_plane(x0, x1, t, w):
    y = w[0] * x0 + w[1] * x1 + w[2]
    mse = np.mean((y-t)**2)
    return mse

# 메인
plt.figure(figsize=(6,5))
ax = plt.subplot(1,1,1,projection='3d')
W = [1.5, 1, 90] # 평면의 방정식: 매개변수, w를 움직이면 면이 여러방향을 향하며, 평균제곱오차함수가 변함.
show_plane(ax, W)
show_data2(ax, X0, X1, T)
mse = mse_plane(X0, X1, T, W)
print("SD={0:.3f} cm".format(np.sqrt(mse)))
plt.show()