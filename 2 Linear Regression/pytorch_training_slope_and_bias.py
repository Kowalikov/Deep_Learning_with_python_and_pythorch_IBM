import numpy as np
import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt
import torch

def forward(x):
    return w * x + b

def criterion(yhat, y):
    return torch.mean((yhat - y) ** 2)

class plot_diagram():

    # Constructor
    def __init__(self, X, Y, w, b, stop, go=False):
        startw = w.data
        startb = b.data
        self.error = []
        self.parameter = []
        self.X = X.numpy()
        self.Y = Y.numpy()
        self.parameter_values = torch.arange(startw, stop)
        self.Loss_function = [criterion(forward(X), Y) for w.data in self.parameter_values]
        w.data = startw

    # Executor
    def __call__(self, Yhat, w, error, n):
        self.error.append(error)
        self.parameter.append(w.data)
        plt.subplot(212)
        plt.plot(self.X, Yhat.detach().numpy())
        plt.plot(self.X, self.Y, 'ro')
        plt.xlabel("A")
        plt.ylim(-20, 20)
        plt.subplot(211)
        plt.title("Data Space (top) Estimated Line (bottom) Iteration " + str(n))
        plt.plot(self.parameter_values.numpy(), self.Loss_function)
        plt.plot(self.parameter, self.error, 'ro')
        plt.xlabel("B")
        plt.show()

    # Destructor
    def __del__(self):
        plt.close('all')


def train_model(iter):
    for epoch in range(iter):
        # make the prediction as we learned in the last lab
        Yhat = forward(X)

        # calculate the iteration
        loss = criterion(Yhat, Y)

        # plot the diagram for us to have a better idea
        if epoch == iter-1:
            gradient_plot(Yhat, w, loss.item(), epoch)

        # store the loss into list
        LOSS.append(loss)

        print("Loss ", epoch, "#: ", loss, sep="")
        print("W:", w.data, "\tb:", b.data)

        # backward pass: compute gradient of the loss with respect to all the learnable parameters
        loss.backward()

        # updata parameters
        w.data = w.data - lr * w.grad.data
        w.grad.data.zero_()

        b.data = b.data - lr * b.grad.data
        b.grad.data.zero_()

X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = -3 * X - 1
Y = f + 0.2 * torch.randn(X.size())

lr = 0.1
LOSS = []
w = torch.tensor(-10.0, requires_grad = True)
b = torch.tensor( 5.0, requires_grad = True)
gradient_plot = plot_diagram(X, Y, w, b, stop = 5)

train_model(12)
plt.plot(LOSS)
plt.tight_layout()
plt.xlabel("Epoch/Iterations")
plt.ylabel("Cost")
plt.show()

X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = 1 * X - 1
Y = f + 0.1 * torch.randn(X.size())
