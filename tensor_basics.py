import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

b=[]
for x in range(10):
    b.append(float(x))

a=torch.tensor(b) #tworzenie tensora z listy
print(a.dtype) #typ danych z tensora
print(a.type()) #typ tensora, (bo tensory też mają typy)
a2 = a.type(torch.DoubleTensor) #zmiana typu tensora, działa tylko dla nowych tensorów
print(a2.type())

print(a.size()) #ilośc elementów tensora
print(a.ndimension()) #wymiar tensora
a_col=a.view(-1,1) #widok macierzowy, chyba transpozycja, -1 bo pozycja ostatniego elementu jako rozmiar w rzędach
print(a_col)

numpy_array = np.linspace(0, 10, 5, endpoint=False) #numpy array creation with linspace function, range and amount of numbers to create
torch_tensor = torch.from_numpy(numpy_array) #create a tensor made from an numpy array
back_to_numpy = torch_tensor.numpy() #create a numpy array out of a tensor
print("b:", numpy_array, torch_tensor, back_to_numpy)

pandas_series = pd.Series([0.1, 3, 0.3, 4, 0, 10]) #defining pandas' type series
torch_tens = torch.from_numpy(pandas_series.values)
print(torch_tens)

this_tensor = torch.tensor([0,1,2,3]) #list converting
torch_to_list = this_tensor.tolist()
print(torch_to_list)

print(this_tensor[0]) #print raw data with tensor structure
print(this_tensor[-1])
print(this_tensor[0].item()) #print tensor value
print(this_tensor[-1].item())

c = torch.tensor([1,2,3,4]) #imprinting part of the tensor
d = c[1:3]
c[2:4] = torch.tensor([300, 500])
print(d, c) #why copied values still depand on the previous values?

e = d+c[2:4]
print(e)
e = d*c[2:4] #simple product of tensor
print(e)
e = torch.dot(d, c[2:4]) #dot product of the two tensors
print(e)
e = d+1 #add a constant to the tensor - "broadcasting"
print(e)

print(c.float().mean().item()) #średnia dostępna tylko dla floatów i to zwracana jako 0D tensor
print(c.max().item())

f = torch.sin(torch.from_numpy(np.linspace(0, np.pi*10, 20, endpoint=False)).clone().detach()).long() #oneliner flex
print(f)


gx = torch.linspace(0, np.pi*2, 101) #previous exaple simplier assign
gy = torch.sin(gx)
plt.plot(gx.numpy(), gy.numpy())
plt.show()





#print('a:', a, a.dtype)


