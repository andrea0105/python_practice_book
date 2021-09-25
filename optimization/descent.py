import sympy as sp
import numpy as np

x = sp.Symbol('x')
y = sp.Symbol('y')

f = 50 * (y - x**2)**2 + (2 - x)**2

c = sp.derive_by_array(f, (x, y))
Hf = sp.derive_by_array(c, (x, y))
first_ncs_sol = sp.solve(c)
H = Hf.subs({x:first_ncs_sol[0][x], y:first_ncs_sol[0][y]})

H_ = np.array(H).astype(np.float32).reshape(2,2)
lamda, _ = np.linalg.eig(H_)
print(lamda, _)