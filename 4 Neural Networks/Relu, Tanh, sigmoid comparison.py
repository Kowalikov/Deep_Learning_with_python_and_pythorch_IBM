import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("macosx")

torch.manual_seed(2)
z = torch.arange(-10, 10, 0.1).view(-1, 1)
sig = nn.Sigmoid()
yhat = sig(z)
plt.plot(z.numpy(),yhat.numpy())
plt.xlabel('z')
plt.ylabel('yhat')
plt.show()
yhat = torch.sigmoid(z)
plt.plot(z.numpy(), yhat.numpy())
plt.show()

TANH = nn.Tanh()
yhat = TANH(z)
plt.plot(z.numpy(), yhat.numpy())
plt.show()
yhat = TANH(z)
plt.plot(z.numpy(), yhat.numpy())
plt.show()

RELU = nn.ReLU()
yhat = RELU(z)
plt.plot(z.numpy(), yhat.numpy())
yhat = F.relu(z)
plt.plot(z.numpy(), yhat.numpy())
plt.show()

x = torch.arange(-2, 2, 0.1).view(-1, 1)
plt.plot(x.numpy(), F.relu(x).numpy(), label='relu')
plt.plot(x.numpy(), torch.sigmoid(x).numpy(), label='sigmoid')
plt.plot(x.numpy(), torch.tanh(x).numpy(), label='tanh')
plt.legend()