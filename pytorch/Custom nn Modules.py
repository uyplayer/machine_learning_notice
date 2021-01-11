#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/3 19:49
# @Author  : uyplayer
# @Site    : uyplayer.pw
# @File    : Custom nn Modules.py
# @Software: PyCharm

# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-custom-nn-modules

'''
Sometimes you will want to specify models that are more complex than a sequence of existing Modules; for these cases you can define your own Modules by subclassing nn.Module and defining a forward which receives input Tensors and produces output Tensors using other modules or other autograd operations on Tensors.
'''

# dependency library
# pytorch
import torch

# we are going to use Custom nn Modules,because sequence model is really simple
class TwoLayerNet(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    # input
    def forward(self,x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# random
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# model
model = TwoLayerNet(D_in, H, D_out)

# loss fucntion
criterion = torch.nn.MSELoss(reduction='sum')

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

# training
for t in range(500):

    # input
    y_pred = model(x)

    # loss
    loss = criterion(y_pred,y)

    if t % 100 == 99:
        print(t, loss.item())

    # grad
    optimizer.zero_grad()

    # compute gradient of the loss with respect to model
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()



