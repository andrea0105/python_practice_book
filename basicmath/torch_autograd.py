import numpy as np
import torch

x = torch.tensor([1.0], requires_grad=True)
y = torch.tensor([2.0], requires_grad=True)
f_xy = (x**2 + 2*x) * torch.log(y)

torch.autograd.backward(f_xy, retain_graph=True)
df = torch.autograd.grad(f_xy, (x, y), retain_graph=True)
dz = torch.autograd.grad(f_xy, (x, y), grad_outputs=torch.tensor([1.]), retain_graph=True)
print(dz)