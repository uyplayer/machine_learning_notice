#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/27 15:09
# @Author  : uyplayer
# @Site    : uyplayer.pw
# @File    : test.py
# @Software: PyCharm

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w

print(conv_output_shape([15, 15], 5, stride=1, pad=0, dilation=1))