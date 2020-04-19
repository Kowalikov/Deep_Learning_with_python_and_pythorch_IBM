import torch
from torch.nn import Linear

torch.manual_seed(1)

w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(-1.0, requires_grad=True)

def forward(x):
    y=w*x+b
    return y

x = torch.tensor([[4.0], [2.0]])

yhat = forward(x) #made by hand model

z = torch.mul(yhat.T, yhat) #testing matrix multiples with transpose option
print(z, yhat.T, yhat)

model = Linear(in_features=1, out_features=1) #imported linear model
print(list(model.parameters()))

yhat = model(x)
print(yhat)
