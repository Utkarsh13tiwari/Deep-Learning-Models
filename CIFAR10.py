import torch
import torch.nn as nn
import torchvision.datasets
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import moviepy.editor
import librosa
import numpy as np
import os
import random
from torchvision.transforms import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Flow of the program : Forward pass , Loss computation , Backward pass , Update weights
# data loading
# 2) model building
# 3) iterations and training

n_epoch = 200
l_rate = 0.01
batch_size = 4

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)

dataiter = iter(train_loader)

#images, labels = dataiter.next()

#plt.imshow(images)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def conv_shape(x, k=1, p=0, s=1, d=1):
    #return int((x + 2*p - d*(k - 1) - 1)/s + 1)
     return int((((x-k) + 2*p)/s)+1)
# ((W - F) + 2P ) / S) + 1
w = conv_shape(conv_shape(16, k=5, p=0), k=5, s=1)
w = conv_shape(w, k=5, s=1)

print(w)

class convnn(nn.Module):
    def __init__(self):
        super(convnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 16, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 32, 5, 5
        #x = x.view(-1, 32 * 5 * 5)  # -> n, 32*5*5
        x= x.view(-1,32*5*5)
        x = F.relu(self.fc1(x))  # -> n, 120
        x = F.relu(self.fc2(x))  # -> n, 84
        x = self.fc3(x)  # -> n, 10
        return x


model = convnn().to(device)
criteration = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

n_total_steps = len(train_loader)

for epoch in range(n_epoch):
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        #forward pass
        output = model(images)
        #loss calculatioin
        loss = criteration(output,labels)

        #backward
        optimizer.zero_grad() # deleting old grad
        loss.backward() # backward
        optimizer.step()

    if (i + 1) % 2000 == 0:
        print(f'Epoch [{epoch + 1}/{n_epoch}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
