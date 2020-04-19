#PSEUDOCODE

#two ways of implementing the neural network model
import torch.functional as F
import torch


class Net(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear(D_in, H)
        self.linear(H, D_out)
    def forward(self, x):
        x = F.sigmoid(self.linear1(x))
        x = F.sigmoid(self.linear2(x))
        return x


x = torch.linspace(-3.0, 3.0, 0.1, requires_grad=True)
model = Net(1,2,1) #model with one input, one hidden layer with two hidden neurons, one output layer
#alternatively
model = torch.nn.Sequential(torch.nn.Linear(1,2), torch.nn.Sigmoid(), torch.nn.Linear(2,1), torch.nn.Sigmoid())

yhat = model(x) #prediction

#just another piece:
#when the output as here is logistic, as a creterion we are using the BCEloss
criterion = torch.nn.BCEloss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
train(dataset, model, criterion, train_loader, optimizer, epochs=1000)
