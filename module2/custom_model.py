import torch.nn as nn
import torch

torch.manual_seed(69)

class LR(nn.Module):
    def __init__(self, in_size, out_size):
        super(LR, self).__init__() #alternatywa: nn.Module.__init__(self)
        self.linear=nn.Linear(in_size, out_size)

    def forward(self, x):
        out = self.linear(x)
        return out

model = LR(1, 1)
print(list(model.parameters()))

x = torch.tensor([[4.0], [7.0]])
yhat = model(x)
print("yhat: ", yhat)