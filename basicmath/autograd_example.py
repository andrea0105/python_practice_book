import numpy as np

def times(x, y):
    return x*y, (x, y)

def times_deriv(cache, dout=1):
    return cache[1]*dout, cache[0]*dout

TIMES = {'f': times, 'df': times_deriv}

#v, cache = TIMES['f'](2,3)
#dx, dy = TIMES['df'](cache)

#print("dx={}, dy={}".format(dx, dy))

def add(x, y):
    return x+y, (x,y)

def add_deriv(cache, dout=1):
    return dout, dout

ADD = {'f': add, 'df': add_deriv}

def log(x):
    return np.log(x), x

def log_deriv(cache,dout=1):
    return (1/cache) * dout

LOG = {'f': log, 'df': log_deriv}

x = 1.; y = 2

# F(x, y) = (x**2 + 2*x) * np.log(y)
# 캐쉬는 입력
a, cache_a = TIMES['f'](x, x)
print(cache_a)
b, cache_b = TIMES['f'](2, x)
print(cache_b)
c, cache_c = ADD['f'](a, b)
print(c, cache_c)
d, cache_d = LOG['f'](y)
print(d, cache_d)
z, cache_z = TIMES['f'](c, d)
print(cache_z)
# 위의 함수를 순전파 시키는 순서

dx = dy = 0.
# 밑의 과정은 역전파 과정, 미분계수를 찾는 과정
dc, dd = TIMES['df'](cache_z, 2) # 마지막 곱셈함수의 입력은 c, d 였다. 
#따라서 미분계수는 2개. dd는 다음 로그함수의 상류층 미분계수
print(dc, dd)
dy     = LOG['df'](cache_d, dd)
print(dy)
da, db = ADD['df'](cache_c, dc)
print(da, db)
_, dx_ = TIMES['df'](cache_b, db); dx += dx_;
print(dx)
dx_, dx__ = TIMES['df'](cache_a, da); dx += dx_ + dx__;
print(dx)

print("backward pass dx = {:.6f}, dy = {:.6f}".format(dx, dy))
