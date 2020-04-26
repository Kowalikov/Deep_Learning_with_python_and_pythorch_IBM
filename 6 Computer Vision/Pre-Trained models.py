import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as dataset
import torch.nn as nn
import torch

model = models.resnet18(pretrained=True)


mean = [0.486,  0.456, 0.406]
std =  [0.224, 0.225, 0.229]

transforms_stuff = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean, std)])
train_dataset = dataset( root='./data', download=True, transforms=transforms_stuff) #cokolwiek, byle kolorowy obraz
validation_dataset = dataset( root='./data', splti='test', download=True, transforms=transforms_stuff)

for param in model.parameters():
    param.requires_grad=False

model.fc = nn.Linear( 512,3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([parameters for parameters in model.parameters() if parameters.requires_grad], lr = 0.001)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=100)

for epoch in range(n_epochs):
    for x,y in train_loader:
        optimizer.zero_grad()
        z=model(x)
        loss=criterion(z, y)
        loss.backward()
        optimizer.step()
correct=0

    for x_text, y_test in validation_loader:
        z=model(x)
        _, yhat = torch.max(z.data, 1)
        correct+=yhat==y_test().sum().item()


accuracy=correct/N_test
accuracy_list.append(accuracy)
loss_list.append(loss.data)