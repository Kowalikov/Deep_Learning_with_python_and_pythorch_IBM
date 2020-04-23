import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, misc

conv = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=3)

def simpleConv():
    print("Some simple conv shit:\n")
    conv.state_dict()['weight'][0][0] = torch.tensor([[1.0, 0, -1.0], [2.0, 0, -2.0], [1.0, 0.0, -1.0]])
    conv.state_dict()['bias'][0] = 0.0
    print(conv.state_dict())

    image = torch.zeros(1, 1, 5, 5).uniform_(0, 1)  # image with zeros with one input, one channel, 5 rows,5 columns
    image[0, 0, :, 2] = 1  # setting third column as with 1s
    print(image)

    z = conv(image)
    print(z, "\n\n")

def CustomSizeConvolution(K=2):
    print("Conv avec kernel size like", K, ":")
    conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=K)
    conv1.state_dict()['weight'][0][0] = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    conv1.state_dict()['bias'][0] = 0.0
    conv1.state_dict()
    print(conv1)

    M = 4
    image1 = torch.ones(1, 1, M, M)

    z1 = conv1(image1)
    print("z1:", z1)
    print("shape:", z1.shape[2:4])

def FullyCustomConv(K=2, stride=2):
    conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=K, stride=stride)

    conv3.state_dict()['weight'][0][0] = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    conv3.state_dict()['bias'][0] = 0.0
    conv3.state_dict()

    M = 4
    image1 = torch.ones(1, 1, M, M)
    z3 = conv3(image1)

    print("z3:", z3)
    print("shape:", z3.shape[2:4])

def Padding():
    M = 4
    image1 = torch.ones(1, 1, M, M)
    conv4 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=3)
    conv4.state_dict()['weight'][0][0] = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    conv4.state_dict()['bias'][0] = 0.0
    conv4.state_dict()
    z4 = conv4(image1)
    print("z4:", z4)
    print("z4:", z4.shape[2:4])
    print("Wroung! Yu nid two maek thiz in di otter wei. Liek this:")

    conv5 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=(1,3), padding=1,)
    conv5.state_dict()['weight'][0][0] = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    conv5.state_dict()['bias'][0] = 0.0
    conv5.state_dict()
    z5 = conv5(image1)
    print("z5:", z5)
    print("z5:", z4.shape[2:4])



simpleConv() #see example
CustomSizeConvolution() #notice the size change  Mout=(Min-K)/stride+1
FullyCustomConv()
Padding()