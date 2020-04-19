import torch as torch
import matplotlib.pyplot as plt
import numpy as np

x = torch.tensor(2.0 ,requires_grad=True)
y1 = x**2
y1.backward()
x.grad
print(y1, x.grad)

y2 = x**2+x*2+2
y2.backward()
x.grad
print(y2, x.grad)

y3 = 3*x**3+5
x.grad
x.grad.zero_()
y3.backward(retain_graph = True)

# derivatives of many vaibles function
u = torch.tensor(1.0, requires_grad=True)
v = torch.tensor(2.0, requires_grad=True)
f = u*v + u**2
f.backward()
print(u.grad, v.grad)

x1 = torch.linspace( -10, 10, 2001, requires_grad=True)
y4 = torch.sin(x1)**2
y5 = torch.sum(y4)
y5.backward()



plt.plot( x1.detach().numpy(), y4.detach().numpy(), label='function') #always use a deteach function for tensors which requiers gradients
plt.plot( x1.detach().numpy(), x1.grad.detach().numpy(), label='derivative')
plt.legend()
plt.grid()
plt.show()







