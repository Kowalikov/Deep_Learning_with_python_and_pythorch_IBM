import torch
from torch.utils.data import Dataset
from torchvision import transforms

torch.manual_seed(1)


class toy_set(Dataset):

    # Constructor with defult values
    def __init__(self, length=100, transform=None):
        self.len = length
        self.x = 2 * torch.ones(length, 2)
        self.y = torch.ones(length, 1)
        self.transform = transform

    # Getter
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    # Get Length
    def __len__(self):
        return self.len


class add_mult(object):

    # Constructor
    def __init__(self, addx=1, muly=2):
        self.addx = addx
        self.muly = muly

    # Executor
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x + self.addx
        y = y * self.muly
        sample = x, y
        return sample

class my_add_mult(object):

    def __init__(self, addx=10, addy=10, mulx=8, muly=7):
        self.addx = addx
        self.addy = addy
        self.muly = muly
        self.mulx = mulx

    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = (x +self.addx)*self.mulx
        y = (y + self.addy) * self.muly
        sample = x, y
        return sample

a_m = add_mult()
m_a_m = my_add_mult(addx=2, addy=2, mulx=10, muly=10)
our_dataset = toy_set(length=50) #we can create a dataset on a different length than declared
print("Our toy_set object: ", our_dataset)
print("Value on index 0 of our toy_set object: ", our_dataset[0])
print("Our toy_set length: ", len(our_dataset))

for x in range(3):
    x_, y_ = a_m(our_dataset[x])
    x1, y1 = m_a_m(our_dataset[x])
    print(x, "th element of our dataset after a_m transform: ", x_, " ", y_, sep='')
    print(x, "th element of our dataset after a_m transform: ", x1, " ", y1, sep='')

data_transform = transforms.Compose([add_mult(), my_add_mult()])
compose_data_set = toy_set(length=10, transform=data_transform)

for i in range(3):
    x, y = our_dataset[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_co, y_co = compose_data_set[i]
    print('Index: ', i, 'Compose Transformed x_co: ', x_co ,'Compose Transformed y_co: ',y_co)
