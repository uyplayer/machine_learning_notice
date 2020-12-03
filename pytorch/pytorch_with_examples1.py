#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/3 12:17
# @Author  : uyplayer
# @Site    : uyplayer.pw
# @File    : pytorch_with_examples1.py
# @Software: PyCharm

# dependency library
# data process
import numpy as np
import pandas as pd
# torch
import torch
# model summary
import torchsummary as summary

dtype = torch.float
device = torch.device("cpu")

# N is batch size; D_in is input dimension
# H is hidden dimension; D_out is output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# # Create random Tensors to hold input and outputs
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # torch.mm  Performs a matrix multiplication of the matrices input and mat2
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    '''
    >>> mat1 = torch.randn(2, 3)
    >>> mat2 = torch.randn(3, 3)
    >>> torch.mm(mat1, mat2)
    tensor([[ 0.4851,  0.5037, -0.3633],
            [-0.0760, -3.6705,  2.4784]])
    '''
    # Clamp all elements in input into the range [ min, max ] and return a resulting tensor
    '''
    >>> a = torch.randn(4)
    >>> a
    tensor([-1.7120,  0.1734, -0.0478, -0.0922])
    >>> torch.clamp(a, min=-0.5, max=0.5)
    tensor([-0.5000,  0.1734, -0.0478, -0.0922])
    '''
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # autograd(auto-gred)
    loss.backward()

    # manually update the weights
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w1 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()
