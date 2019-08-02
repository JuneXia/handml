
"""
import torch

# 创建一个5x3的矩阵，但是未初始化
x = torch.empty(5, 3)
print(x)


x = torch.rand(5, 3)
print(x)

arr = x.view(15)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)


x = torch.tensor([5.5, 3])
print(x)


if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
"""


"""
import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)
"""

import torch
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()


a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)


out.backward()
print(x.grad)


x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)



gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)
print(x.grad)



print(x.requires_grad)
print((x ** 2).requires_grad)
with torch.no_grad():
    print((x ** 2).requires_grad)


import torch as t
from torch.autograd import Variable

x = Variable(t.ones(2, 2, requires_grad=True))
y = x.sum()
y.grad_fn

arr = y.backward()

print('debug')



x.long()