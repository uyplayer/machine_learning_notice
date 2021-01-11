#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/20 10:40
# @Author  : uyplayer
# @Site    : uyplayer.pw
# @File    : autograd_tutorial.py
# @Software: PyCharm

import torch
import numpy as np

# xx = torch.ones(2,2)
# print(xx)
# x = torch.ones(2,2,requires_grad=True)
# print(x)
#
# y = x+2
# print(y)
# print(y.grad_fn)
#
# z = y*y*3
# print(z)
# out = z.mean()
# print(out)
#
# a = torch.randn(2, 2)
# a = ((a * 3) / (a - 1))
# print(a.requires_grad)
# a.requires_grad_(True)
# print(a.requires_grad)
# b = (a * a).sum()
# print(b.grad_fn)
# print(x.grad)

# Create a tensor and set requires_grad=True to track computation with it
# x = torch.ones(2,2,requires_grad=True)
# # print(x)
# # y = x + 2
# # print(y)
# # print(y.grad)
# # print(y.grad_fn)
# #
# # z = y * y * 3
# # out = z.mean()
# #
# # print(z, out)
# # # out.backward()
# # print(x.grad)
# #
# # a = torch.randn(2, 2)
# # a = ((a * 3) / (a - 1))
# # print(a.requires_grad)
# # a.requires_grad_(True)
# # print(a.requires_grad)
# # b = (a * a).sum()
# # print(b.grad_fn)


x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3 # 3x**2 + 6x +12
print(z)
out = z.mean() #
out.backward() # 6(x+2) , 有四个元素，所以 6/4 （x+2） = 3/2（x+2）
print(out)
print(x.grad) # d(out)/dx = 3/2（x+2） ;当x=1，则9/2 = 4.5

# Now let’s take a look at an example of vector-Jacobian product:
x = torch.randn(3,requires_grad=True)
y = x*2
while y.data.norm() < 1000:
    y = y * 2
print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)

# print("======================")
# print(x.requires_grad)
# print((x ** 2).requires_grad)
# with torch.no_grad():
#     print((x ** 2).requires_grad)

print("----------------------")
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())