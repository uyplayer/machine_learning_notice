#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/3 19:15
# @Author  : uyplayer
# @Site    : uyplayer.pw
# @File    : optim.py
# @Software: PyCharm

# dependency library
# pytorch
import torch

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# random
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# define model
model = torch.nn.Sequential(torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out),)

# loss
loss_fn = torch.nn.MSELoss(reduction="sum")

# define optimizer
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(500):
    #Forward
    y_pred = model(x)

    #Compute
    loss = loss_fn(y_pred,y)

    # print loss
    if t % 100 == 99:
        print(t, loss.item())

    '''
    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details
    '''
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
