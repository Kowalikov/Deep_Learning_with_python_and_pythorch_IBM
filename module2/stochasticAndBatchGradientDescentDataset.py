import torch
import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self):
        self.x = torch.arange(-3.0, 3.0, 0.1).view(-1, 1)
        self.f = -3*x
        self.y = self.f + 0.1 * torch.randn(self.x.size())
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


def forward(x):
    return w * x + b


def criterion(yhat, y):
    return torch.mean((yhat - y) ** 2)

w = torch.tensor(-15.0, requires_grad=True)
b = torch.tensor(-10.0, requires_grad=True)

X = torch.arange(-3.0, 3.0, 0.1).view(-1, 1)
f = -3*X
Y = f + 0.2*torch.randn(X.size())
lr = 0.1

for epoch in range(6):
    for x,y in zip(X,Y):
        yhat = forward(X)
        loss = criterion(yhat, y)
        loss. backward()
        w.data = w.data - lr * w.grad.data
        b.data = b.data - lr * b.grad.data
        b.grad.data.zero_()
        w.grad.data.zero_()


plt.plot(X.numpy(), f.numpy())
plt.plot(X.numpy(), Y.numpy(), color='red', marker='o', markersize=2, linestyle='None')
#plt.show()

print(w.data, b.data)

dataset = Data()
trainloader = Dataset.DataLoader(dataset=dataset, batch_size=1)

for epoch in range(6):
    for x,y in trainloader:
        yhat = forward(X)
        loss = criterion(yhat, y)
        loss. backward()
        w.data = w.data - lr * w.grad.data
        b.data = b.data - lr * b.grad.data
        b.grad.data.zero_()
        w.grad.data.zero_()