#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/30 14:24
# @Author  : uyplayer
# @Site    : uyplayer.pw
# @File    : pytorch_with_examples.py
# @Software: PyCharm

# dependency library
import numpy as np

# N is batch size; D_in is input dimension
# H is hidden dimension; D_out is output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in) #(64,1000)
y = np.random.randn(N, D_out) #(64,10)

# Randomly initialize weights
w1 = np.random.randn(D_in, H) #(1000,100)
w2 = np.random.randn(100, D_out) #(100,10)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.dot(w1) # (64,1000)*(1000,100) = (64,100)
    h_relu = np.maximum(h,0) # if an element in array is small than 0 ; then 0 replaces this element
    y_pred = h_relu.dot(w2) # (64,100)*(100,10) = (64,10)

    #loss
    loss = np.square(y_pred - y) # (64,10)
    loss = loss.sum() # a float number
    print(t, loss)

