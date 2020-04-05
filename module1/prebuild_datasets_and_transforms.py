import torch
import matplotlib.pylab as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dsets


def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + str(data_sample[1].item()))

torch.manual_seed(0)

croptensor_data_transform = transforms.Compose([transforms.CenterCrop(20), transforms.ToTensor()])
dataset = dsets.MNIST(root = './data', train = False, download=True, transform = croptensor_data_transform) #nie pobiera się
print("The shape of the first element in the first tuple: ", dataset[0][0].shape)

show_data(dataset[0],shape = (20, 20))
show_data(dataset[1],shape = (20, 20))

fliptensor_data_transform = transforms.Compose([transforms.RandomHorizontalFlip(p = 1),transforms.ToTensor()])
dataset = dsets.MNIST(root = './data', train = False, download = True, transform = fliptensor_data_transform)
show_data(dataset[1])