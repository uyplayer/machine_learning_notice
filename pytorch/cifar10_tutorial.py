#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/27 10:28
# @Author  : uyplayer
# @Site    : uyplayer.pw
# @File    : cifar10_tutorial.py
# @Software: PyCharm

# dependency library
# Pytorch
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
# torchsummary
from torchsummary import summary # model summary
# numirical
import matplotlib.pyplot as plt
import numpy as np

# to normalize image data
'''
transforms.Compose receive a list object; 
Normalize does the following for each channel:
image = (image - mean) / std
The parameters mean, std are passed as 0.5, 0.5 in your case. This will normalize the image in the range [-1,1]. For example, the minimum value 0 will be converted to (0-0.5)/0.5=-1, the maximum value of 1 will be converted to (1-0.5)/0.5=1.
if you would like to get your image back in [0,1] range, you could use,
image = ((image * std) + mean)
About whether it helps CNN to learn better, I’m not sure. But majority of the papers I read employ some normalization schema. What you are following is one of them.
mean = (0.5, 0.5, 0.5), std(0.5, 0.5, 0.5)
std =sqrt(((x1-x)^2 +(x2-x)^2 +......(xn-x)^2)/n) ; n is number of sample ; x is mean value of smaple 
'''
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# load training data
trainset = torchvision.datasets.CIFAR10(root = "./data",train=True,download=True,transform=transform)
# training  loader
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True, num_workers=0)
# load test data
testset = torchvision.datasets.CIFAR10(root = "./data", train=False,download=True, transform=transform)
# test  loader
testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False, num_workers=0)
# image classes
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# functions to show an image
def imshow(img):
    # unnormalize If z=(x-mu)/s, then x=z*s+mu
    img = img/2+0.5
    npimg = img.numpy()
    '''
    Reverse or permute the axes of an array; returns the modified array
    
    x = np.arange(4).reshape((2,2))
    array([[0, 1],
       [2, 3]])
    np.transpose(x)
    array([[0, 2],
       [1, 3]])
    x = np.ones((1, 2, 3))
    np.transpose(x, (1, 0, 2)).shape
    (2, 1, 3)
    '''
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader) #
images, labels = dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images)) # Make a grid of images
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# Define a Convolutional Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # NetWork structure and the upper and lower order of these lines are nothong to do with calculation,
        # these lines are definatiom of the fuctions
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # calculation is here , input image size : (32,32)
        x = self.conv1(x) # (28, 28)
        x = self.pool(F.relu(x)) # (14, 14)
        x = self.conv2(x) # (10, 10)
        x = self.pool(F.relu(x)) # (5, 5)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# init model
net = Net()
# output the model architecture
# image 32*32
summary(net, (3, 32, 32))








