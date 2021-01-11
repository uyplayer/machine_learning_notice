#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/3 17:55
# @Author  : uyplayer
# @Site    : uyplayer.xyz
# @File    : Underfit-Overfit.py
# @Software: PyCharm

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

X = np.arange(1,16)
y = np.array([7, 8, 7, 13, 16, 15, 19, 23, 18, 21]).reshape(10, 1)
y = np.append(y, [24, 23, 22, 26, 22])

plt.plot(X,y,'ro')
plt.show()
