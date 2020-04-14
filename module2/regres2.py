import random
import torch
from torch.nn import Linear
import matplotlib.pyplot as plt

plt.interactive(True)
w = torch.tensor(10.0, requires_grad=True)
X = torch.arange(-3, 3, 0.1).view(-1,1)
f =-3*X

plt.plot(X.numpy(), f.numpy())
plt.show()

Y = f + 0.1*torch.randn(X.size())

plt.plot(X.numpy(), Y.numpy(), 'ro')
plt.show()

def forward(x):
    y = w*x
    return y

def criterion(yhat, y):
    return torch.mean((yhat-y)**2)


lr = 0.1
LOSS = []

for epoch in range(4):
    Yhat = forward(X)
    loss = criterion(Yhat,Y)
    loss.backward()
    w.data = w.data-lr*w.grad.data
    w.grad.data.zero_()
    LOSS.append(loss)
    

