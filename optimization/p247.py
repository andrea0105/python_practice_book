import numpy as np
from scipy.optimize import line_search

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

# 시작점 x^(0)
X = np.array([-1, 2])

# 수렴상수 epsilon 설정
def SDM(f, df, x, eps=1.0e-7, callback=None):
    max_iter = 10000

    # 단계 1. 반복횟수 k=0으로 설정
    for k in range(max_iter):
        # 단계 2. 경사도 벡터 계산 : c^(k) = 델f(x^(k))
        c = df(x)

        # 단계 3. 수렴판정: c^(k) < eps 이면 x^*=x^(k) or 진행
        if np.linalg.norm(c) < eps:
            print("Stop criterion break Iter.: {:5d} \
               , x: {}".format(k, x))
            break

        # 단계 4. 강하방향 설정: d^(k) = c^(k)
        d = -c

        # 단계 5.  이동거리 계산: d^(k)를 따라 
        # f(a)=f(x^(k)+a*d^(k)) 를  최소화하는 a_k를 계산
        alpha = line_search(f, df, x, d)[0]

        # 단계 6. 업데이트: x^(k+1)=x^(k)+a_k*d^(k)
        # 로 변수를 업데이트 하고 k=k+1 로 두고 2번으로 가서 반복

        x = x + alpha * d

        if callback:
            callback(x)

        else:
            print("Stop Max iter: {:5d} x:{}".format(k, x))

F1 = f2
DF1 = df2
print(SDM(F1, DF1, X))