import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("macosx")
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class Data(Dataset):
    def __init__(self, N_SAMPLES=40, noise_std=1, train=True):

        self.x = torch.linspace(-1, 1, N_SAMPLES).view(-1, 1)
        self.f = self.x ** 2

        if train != True:
            torch.manual_seed(1)

            self.y = self.f + noise_std * torch.randn(self.f.size())
            self.y = self.y.view(-1, 1)
            torch.manual_seed(0)
        else:
            self.y = self.f + noise_std * torch.randn(self.f.size())
            self.y = self.y.view(-1, 1)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

    def plot(self):
        plt.figure(figsize=(6.1, 10))
        plt.scatter(self.x.numpy(), self.y.numpy(), label="Samples")
        plt.plot(self.x.numpy(), self.f.numpy(), label="True function", color='orange')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim((-1, 1))
        plt.ylim((-2, 2.5))
        plt.legend(loc="best")
        plt.show()

data_set=Data()
data_set.plot()
torch.manual_seed(0)
validation_set=Data(train=False)

torch.manual_seed(4)
hl_n = 30
model = nn.Sequential(nn.Linear(1, hl_n), nn.ReLU(), nn.Linear(hl_n, hl_n), nn.ReLU(), nn.Linear(hl_n, 1))
model_drop = nn.Sequential(nn.Linear(1, hl_n), nn.Dropout(0.5), nn.ReLU(), nn.Linear(hl_n, hl_n), nn.Dropout(0.5), nn.ReLU(), nn.Linear(hl_n, 1))

model_drop.train()
optimizer_ofit = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer_drop = torch.optim.Adam(model_drop.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

LOSS={}
LOSS['training data no dropout']=[]
LOSS['validation data no dropout']=[]
LOSS['training data dropout']=[]
LOSS['validation data dropout']=[]
epochs = 500

for epoch in range(epochs):
    # make a prediction for both models
    yhat = model(data_set.x)
    yhat_drop = model_drop(data_set.x)
    # calculate the lossf or both models
    loss = criterion(yhat, data_set.y)
    loss_drop = criterion(yhat_drop, data_set.y)

    # store the loss for  both the training and validation  data for both models
    LOSS['training data no dropout'].append(loss.item())
    LOSS['validation data no dropout'].append(criterion(model(validation_set.x), validation_set.y).item())
    LOSS['training data dropout'].append(loss_drop.item())
    model_drop.eval()
    LOSS['validation data dropout'].append(criterion(model_drop(validation_set.x), validation_set.y).item())
    model_drop.train()

    # clear gradient
    optimizer_ofit.zero_grad()
    optimizer_drop.zero_grad()
    # Backward pass: compute gradient of the loss with respect to all the learnable parameters
    loss.backward()
    loss_drop.backward()
    # the step function on an Optimizer makes an update to its parameters
    optimizer_ofit.step()
    optimizer_drop.step()

model_drop.eval()

#Test the accuracy of the model without dropout on the validation data.
_,yhat=torch.max(model(validation_set.x),1)
(yhat==validation_set.y).numpy().mean()
#with drop
_,yhat=torch.max(model_drop(validation_set.x),1)
(yhat==validation_set.y).numpy().mean()

plt.figure(figsize=(6.1, 10))

plt.scatter(data_set.x.numpy(), data_set.y.numpy(), label="Samples")
plt.plot(data_set.x.numpy(), data_set.f.numpy()  ,label="True function",color='orange')
plt.plot(data_set.x.numpy(),yhat.detach().numpy(),label='no dropout',c='r')
plt.plot(data_set.x.numpy(),yhat_drop.detach().numpy(),label="dropout",c='g')


plt.xlabel("x")
plt.ylabel("y")
plt.xlim((-1, 1))
plt.ylim((-2, 2.5))
plt.legend(loc="best")
plt.show()

plt.figure(figsize=(6.1, 10))
for key, value in LOSS.items():
    plt.plot(np.log(np.array(value)),label=key)
    plt.legend()
    plt.xlabel("iterations")
    plt.ylabel("Log of cost or total loss")