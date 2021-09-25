import numpy as np
from scipy import optimize


np.random.seed(3)

# 0~5에서 무작위 점 500개 추출
N = 500
samples = (np.random.rand(N*2)*5).reshape(N,2)

dcs_bnd_1 = lambda x: -(3/5)*x + 3
dcs_bnd_1_imp = lambda x, y: (3)*x + (5)*y + (-15)

dcs_bnd_2 = lambda x: -2*x + 6
dcs_bnd_2_imp = lambda x, y: (-6)*x + (-3)*y + (18)

y1_bin = dcs_bnd_1_imp(samples[:,0], samples[:,1]) > 0
y2_bin = dcs_bnd_2_imp(samples[:,0], samples[:,1]) < 0

positive_where = np.where((y1_bin | y2_bin))[0]
target = np.zeros(N)
target[positive_where] = 1

def sigmoid(x):
    return 1 / ( 1+np.exp(-x))
def network(X, W):
    '''
    X : (N, D) N은 데이터의 수, D는 데이터의 차원수,
    N은 400 D는 2
    W : (3, 3)
        [b^(1)_1, w^(1)_11, w^(1)_22]
        12
        21
    가중치 9개가 (3,3)행렬로 저장된 형태

    ret : (N,)
    D, H, A = 2, 2, 1
    '''
    X = np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))
    Z1 = np.dot(W[:2,:], X.T)
    A1 = sigmoid(Z1)

    A1 = np.vstack((np.ones(A1.shape[1]), A1))
    Z = np.dot(W[-1,:], A1)
    A = sigmoid(Z)

    return A

W = np.array([ [-15, 3, 5], [18, -6, -3], [4, 10, -9] ])
pred = network(samples, W)
pred[pred>=0.5] = 1
pred[pred<0.5] = 0
result = pred==target

res = np.size(result) - np.count_nonzero(result)
print(res)

def J(W, X, T):
    '''
    W: 가중치 (9,)
    X: 주어진 점 데이터 X: (N,D) 형 어레이
    T: 데이터에 대한 클래스 T, 0 또는 1, T: (N,)
    '''
    N = X.shape[0]
    W = W.reshape(3, 3)

    Y = network(X, W)
    return (1/(2*N)) * ((T-Y)**2).sum()

W_star = optimize.fmin_cg(J, W, args=(samples, target), gtol=1e-06)
w_star = W_star.reshape(3, 3)
print(w_star)
W = w_star
print(res)