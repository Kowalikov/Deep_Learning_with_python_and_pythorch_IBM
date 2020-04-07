import random
import torch
from torch.nn import Linear
import matplotlib.pyplot as plt

w = torch.tensor(10.0, requires_grad=True)
X = torch.arange(-3, 3, 0.1).view(-1,1)
f =-3*X

plt.plot(X.numpy(), f.numpy())
plt.show()

Y = f+ 0.1*torch.randn(X.size())

plt.plot(X.numpy(), Y.numpy(), 'ro')