import sympy as sp

x1 = sp.Symbol('x1')
x2 = sp.Symbol('x2')

f = -sp.Rational(1, 3) * x1**2 * sp.E**x2

H = sp.derive_by_array(sp.derive_by_array(f, (x1, x2)), (x1, x2))
print(H)

print(H.subs({x1:2, x2:0}))