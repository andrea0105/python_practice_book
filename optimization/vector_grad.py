import numpy as np
import torch

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

def logistic(x):
    return 1 / (1 + np.exp(-x))

def forward(X, W, T, retopt='all'):
    '''
    네트워크를 피드 포워드 시킨다. numpy 버전
    X: 네트워크의 입력벡터 shpae:(N,2)
    retopt: 네트워크가 순전파 되면서 각 레이어에서 계산된 결과 값을
    되돌릴 방법을 설정한다.
        -'all': 모든 층에서 계산된 결과를 튜플 형태로 되돌린다.
        -'fval': 함수의 최종출력값만 되돌린다.
    '''
    N = X.shape[0]

    H1 = np.dot(W[:2,1:], X.T)
    Z1 = H1 + W[:2,0].reshape(-1, 1)
    A1 = logistic(Z1)

    H2 = np.dot(W[2, 1:], A1)
    Z2 = H2 + W[2,0]
    A2 = logistic(Z2)

    C = (1/(2*N)) * ((T-A2)**2).sum()

    if retopt == 'all':
        return (H1, Z1, A1, H2, Z2, A2, C)

    elif retopt == 'fval':
        return C
    
def forward_torch(X, W, T, retopt='all'):
    '''
    네트워크를 피드포워드시킨다. 파이토치 버전
    X: 네트워크의 입력벡터 size:(N,2)
    retopt: 네트워크가 순전파되면서 각 레이어에서 계산된 결과 값을 
    되돌릴 방법을 설정한다.
        -'all': 모든 층에서 계산된 결과를 튜플 형태로 되돌린다.
        -'fval': 함수의 최종 출력값만 되돌린다.
    '''
    N = X.size()[0]
    T = torch.tensor(T, dtype=torch.double)

    # 계산결과 검증을 위해 pytorch를 사용하므로 numpy어레이
    # 뿐만이 아니라 pytorch tensor 형태에 대해서도 동일한 연산을
    # 합니다.

    H1 = torch.mm(W[:2,1:], torch.t(X)) # np.dot 대신 torch.mm
    Z1 = H1 + W[:2, 0].view(-1, 1) # reshape() 대신 view()
    A1 = torch.sigmoid(Z1)

    H2 = torch.mm(W[:2,1:], A1)
    Z2 = H2 + W[2,0]
    A2 = torch.sigmoid(Z2)

    C = (1/(2*N)) * ((T-A2)**2).sum()

    if retopt == 'all':
        return (H1, Z1, A1, H2, Z2, A2, C)
    elif retopt == 'fval':
        return C

def print_tensor(t):
    '''
    텐서형 자료를 보기 좋게 프린트하기 위한 보조 함수
    '''
    def namestr(obj, namespace):
        return [name for name in namespace if namespace[name]\
            is obj]
    var_name = namestr(t, globals())[0]

    print("{}:{},{}".format(var_name, t.shape, t.dtype))
    print(t)
    print("-------------------------------------------")

np.random.seed(17)
W = np.random.randn(9).reshape(3, 3)
W_torch = torch.tensor(W, dtype=torch.double)
W_torch.requires_grad=True
# print_tensor(W)
# print_tensor(W_torch)
# print(J(W, samples, target))

# print(C)
N = 1
x = samples[[0]]
x_torch = torch.tensor(x, dtype=torch.double)
x_torch.requires_grad=True
t = target[[0]]

dA2 = -(t-A2)/N