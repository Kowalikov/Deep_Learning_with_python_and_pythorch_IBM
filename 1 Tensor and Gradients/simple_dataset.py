import torch
from torch.utils.data import Dataset
from torchvision import transforms

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

class multi(object):
    def __init__(self, mulx=2, muly=3):
        self.mulx=mulx
        self.muly=muly

    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x*self.mulx
        y = y*self.muly
        sample = x, y
        return sample



dataset = ToySet()
print(len(dataset))
x1, y1 = dataset[0]
print("Raw elements: ", x1, y1)

a_m = add_mult()
print("Raw dataset: ", dataset[0])
x_, y_ = a_m(dataset[0])
print("a_m: ", x_, y_)

m_m = multi()
x_, y_ = m_m(dataset[0])
print("m_m: ", x_, y_)

x_, y_ = m_m(a_m(dataset[0])) #ręczna lista Wengerta
print("m_m(a_m): ", x_, y_)
x_, y_ = a_m(m_m(dataset[0]))
print("a_m(m_m): ", x_, y_)

data_transform = transforms.Compose([add_mult(), multi()])
x_, y_ = data_transform(dataset[0])
print("m_m(a_m) (Composed by torchvision): ", x_, y_)
data_set_tr = ToySet(transform=data_transform)
print(data_set_tr[0])



