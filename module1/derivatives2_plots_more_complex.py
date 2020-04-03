import torch as torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

x = torch.linspace( -3, 3, 201, requires_grad=True)
Y = F.relu(x)
y = torch.sum(Y)
y.backward()



plt.plot( x.detach().numpy(), Y.detach().numpy(), label='function') #always use a deteach function for tensors which requiers gradients
plt.plot( x.detach().numpy(), x.grad.detach().numpy(), label='derivative')
plt.legend()
plt.grid()
plt.show()







