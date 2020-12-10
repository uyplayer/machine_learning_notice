#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/10 14:33
# @Author  : uyplayer
# @Site    : uyplayer.pw
# @File    : pytorch_with_examples3.py
# @Software: PyCharm

# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

# dependency library
import numpy as np
import torch
import math

# create random input and out
'''
x: array([-3.14159265, -3.13844949, -3.13530633, ...,  3.13530633,
        3.13844949,  3.14159265])
x.shape:(2000,)        
'''
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# Randomly initialize weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    # y = a + b x + c x^2 + d x^3
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # loss
    loss = np.square(y_pred-y).sum()
    if t % 100 == 99:
        print(t, loss)

        # Backprop to compute gradients of a, b, c, d with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred * x).sum()
        grad_c = (grad_y_pred * x ** 2).sum()
        grad_d = (grad_y_pred * x ** 3).sum()

        # Update weights
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d
print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')
