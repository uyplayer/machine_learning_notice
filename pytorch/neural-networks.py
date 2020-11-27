#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/25 15:11
# @Author  : uyplayer
# @Site    : uyplayer.pw
# @File    : neural-networks.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        '''
        for an input shape (N,C_in ,H_in ,W_in) ; N:batch_size;C_in:number of input channel ; and H_in ,W_in are input size like a 
        image 8*8
        then this input pass the nn.Conv2d preduce output (N,C_out ,H_out ,W_out);N:batch_size;C_out:output 
        channel;H_out ,W_out;  where  H_out and W_out must be calculate like that  H_out  = (H_in+2×padding[
        0]−dilation[0]×(kernel_size[0]−1)/ stride[0])+1  ;   W_out = (H_in+2×padding[
        0]−dilation[1]×(kernel_size[1]−1)/ stride[1])+1
        nn.Conv2d params below
        in_channels (int) – Number of channels in the input image
        out_channels (int) – Number of channels produced by the convolution
        kernel_size (int or tuple) – Size of the convolving kernel
        stride (int or tuple, optional) – Stride of the convolution. Default: 1
        padding (int or tuple, optional) – Zero-padding added to both sides of the input. Default: 0
        padding_mode (string, optional) – 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
        dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
        groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional) – If True, adds a learnable bias to the output. Default: True
        '''
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = F.max_pool2d(F.relu(x), (2, 2))
        # print(x.shape)
        # If the size is a square you can only specify a single number
        x = self.conv2(x)
        # print(x.shape)
        x = F.max_pool2d(F.relu(x), 2)
        # print("look : ",x.shape)
        x = x.view(-1, self.num_flat_features(x))
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = F.relu(self.fc2(x))
        # print(x.shape)
        x = self.fc3(x)
        # print(x.shape)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
# print(" net :",net)
'''
Net(
  (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=576, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
'''
# print(" net.parameters() : ",net.parameters())
# params = list(net.parameters())
# # print(" params : ",params)
# print(" len(params) : ",len(params))
# print(" params[0].size() : ",params[0].size())

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update

output = net(input)
net.zero_grad()
output.backward(torch.randn(1, 10))
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)