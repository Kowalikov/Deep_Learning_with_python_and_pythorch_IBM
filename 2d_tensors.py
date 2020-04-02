import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def diagonala( tensor2d):
    x = tensor2d.size()
    if x[0]!=x[1]:
        return -1
    for x in range (x[0]):
        print("[", x , ",", x, "]: ", tensor2d[x, x], sep='')
    return 0

a = [ [1,2,3], [1,1,2], [1,2,4]] #basic 2d tensor initialization
A = torch.tensor(a)
c = A.size()
print(A, A.ndimension(), A.shape)

diagonala(A) #index testing function

B = 2*A
C = torch.mm(A, B)
C[1:3, 1] = 0
print(C)

a = np.lin


