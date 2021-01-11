#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/10 20:22
# @Author  : uyplayer
# @Site    : uyplayer.pw
# @File    : nn_tutorial.py
# @Software: PyCharm

# https://pytorch.org/tutorials/beginner/nn_tutorial.html

# dependency library
import numpy as np
import pandas as pd
import math
import torch
import torchsummary

# using datasets tools
from torch.utils.data import TensorDataset,DataLoader

from pathlib import Path
import requests
import pickle
import gzip


DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

'''
print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)
(50000, 784)
(50000,)
(10000, 784)
(10000,)
'''

bs = 64  # batch size

# using TensorDataset tool to iter for train and label
train_ds = TensorDataset(x_train,y_train)

# using batch size
train_dl = DataLoader(train_ds, batch_size=bs)

for xb,yb in train_dl:
    # pass
    print(xb.shape)
    print(yb.shape)
    exit()