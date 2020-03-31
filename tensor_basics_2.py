import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plotVec(vectors):
    ax = plt.axes()

    # For loop to draw the vectors
    for vec in vectors:
        ax.arrow(0, 0, *vec["vector"], head_width=0.05, color=vec["color"], head_length=0.1)
        plt.text(*(vec["vector"] + 0.1), vec["name"])

    plt.ylim(-2, 2)
    plt.xlim(-2, 2)


plt.show

old_int_tensor = torch.tensor([0, 1, 2, 3, 4])
new_float_tensor = old_int_tensor.type(torch.FloatTensor)
tensuar = new_float_tensor.type(torch.LongTensor)
print("The type of the new_float_tensor:", new_float_tensor.type())
print("The type of the new_float_tensor:", tensuar.type())

a = torch.linspace(0, 10, 11)
indexes = np.linspace(0, 4, 5) #using variables to pass the indexes
a[indexes]*= 10
print(a)
