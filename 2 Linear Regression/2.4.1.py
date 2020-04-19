from torch.utils.data import Dataset
import torch
import torch.nn as nn

class Data(Dataset):
    def __init__(self):
        self.x=torch.arange(-3.0, 3.0, 0.1)
        self.y=3*X+1
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

class LR(nn.Module):
    def __init__(self, in_size, out_size):
        super(LR, self).__init__()


dataset = Data()
