from torch import nn
import torch


class linear_regression(nn.Module):

    # Constructor
    def __init__(self, input_size, output_size):
        super(linear_regression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    # Prediction function
    def forward(self, x):
        yhat = self.linear(x)
        return yhat


def forward(x):
    yhat = torch.mm(x, w) + b
    return yhat


w = torch.tensor([[2.0], [3.0]], requires_grad=True)

torch.manual_seed(1)


b = torch.tensor([[1.0]], requires_grad=True)


x = torch.tensor([[1.0, 2.0]])
yhat = forward(x)
print("The result of one sample: ", yhat)

X = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
yhat = forward(X)
print("The result many samples: ", yhat)

model = nn.Linear(2, 1)
yhat = model(x)
print("The result from numpy model: ", yhat)

yhat = model(X)
print("The result, many samples, numpy: ", yhat)

model = linear_regression(2, 1)

print("The parameters, our model: ", list(model.parameters()))
print("The parameters, different way: ", model.state_dict())
yhat = model(x)
print("The result of our model: ", yhat)

yhat = model(X)
print("The result, our model, many samples: ", yhat)

X = torch.tensor([[1.0, 1.0, 100.0], [1.0, 2.0, 150.0], [1.0, 3.0, -50]])
model = linear_regression(3,1)
yhat = model(X)
print("Resultd of bigger X", yhat)