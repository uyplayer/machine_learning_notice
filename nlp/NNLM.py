#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/24 22:36
# @Author  : uyplayer
# @Site    : uyplayer.xyz
# @File    : NNLM.py
# @Software: PyCharm

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
'''
torch.autograd provides classes and functions implementing automatic differentiation of arbitrary scalar valued functions. It requires minimal changes to the existing code - you only need to declare Tensor s for which gradients should be computed with the requires_grad=True keyword.
https://pytorch.org/tutorials/beginner/former_torchies/autograd_tutorial_old.html?highlight=autograd
'''

dtype = torch.FloatTensor

