import torch
from torch.utils.data import Dataset


class ToySet(Dataset):
    def __init__(self, length=100, transform=None):
        self.x = 2*torch.ones(length, 2)
        self.y = torch.ones(length, 1)

        self.len = length
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.len

class add_mult(object):
    def __init__(self, addx=1, muly=1):
        self.addx=addx
        self.muly=muly

    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x + self.addx
        y = y*self.muly
        sample = x, y
        return sample



dataset = ToySet()
print(len(dataset))
x1, y1 = dataset[0]
print(x1, y1)
a_m = add_mult()
print(dataset[0])
x_, y_ = a_m(dataset[0])
print(x_, y_)


