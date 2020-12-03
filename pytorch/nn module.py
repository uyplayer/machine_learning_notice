#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/3 15:42
# @Author  : uyplayer
# @Site    : uyplayer.pw
# @File    : nn module.py
# @Software: PyCharm

# dependency library
# data process
import numpy as np
import pandas as pd
# torch
import torch
# model summary
import torchsummary as summary

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# create random Tensor to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# what is torch.nn.Sequential? Sequential layer is also a module which contains other Modules
model = torch.nn.Sequential(torch.nn.Linear(D_in, H),torch.nn.ReLU(),torch.nn.Linear(H, D_out),)

# loss
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(500):
    pass
