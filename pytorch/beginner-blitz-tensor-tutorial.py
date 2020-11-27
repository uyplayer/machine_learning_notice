#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/20 10:10
# @Author  : uyplayer
# @Site    : uyplayer.pw
# @File    : beginner-blitz-tensor-tutorial.py
# @Software: PyCharm

import torch
import numpy as np

# Converting NumPy Array to Torch Tensor
a = np.ones(5)
print(a)
b = a
print(b)
c = torch.from_numpy(a)
print(c)
d = c.numpy()
print(d)

# CUDA Tensors
if torch.cuda.is_available():
    devide = torch.device("cuda")
    y = torch.ones_like(c,devide=devide)
    c = c.to(devide)
    k = y+c
    print(k)


