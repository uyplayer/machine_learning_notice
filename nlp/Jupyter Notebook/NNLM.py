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
import torch.optim as optim
from torch.autograd import Variable
'''
torch.autograd provides classes and functions implementing automatic differentiation of arbitrary scalar valued functions. It requires minimal changes to the existing code - you only need to declare Tensor s for which gradients should be computed with the requires_grad=True keyword.
https://pytorch.org/tutorials/beginner/former_torchies/autograd_tutorial_old.html?highlight=autograd
'''
# type
dtype = torch.FloatTensor

# sentences
sentences = [ "i like dog", "i love coffee", "i hate milk"]
word_list = " ".join(sentences).split() # ['i', 'like', 'dog', 'i', 'love', 'coffee', 'i', 'hate', 'milk']
# print(" ".join(sentences)) i like dog i love coffee i hate milk
word_list = set(word_list)
word_dict = {w:i  for i,w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict)


# NNLM Parameter
n_step = 2 # n-1 in paper
n_hidden = 2 # h in paper
m = 2 # m in paper


def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]] # (n-1)th words
        target = word_dict[word[-1]] # (n)th word

        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch

# Model
class NNLM(nn.Module):

    def __init__(self):
        super(NNLM,self).__init__()
        self.C = nn.Embedding(n_class, m)
        self.H = nn.Parameter(torch.randn(n_step * m, n_hidden).type(dtype))
        self.W = nn.Parameter(torch.randn(n_step * m, n_class).type(dtype))
        self.d = nn.Parameter(torch.randn(n_hidden).type(dtype))
        self.U = nn.Parameter(torch.randn(n_hidden, n_class).type(dtype))
        self.b = nn.Parameter(torch.randn(n_class).type(dtype))

    def forward(self, X):
        X = self.C(X)
        X = X.view(-1, n_step * m)  # [batch_size, n_step * n_class]
        tanh = torch.tanh(self.d + torch.mm(X, self.H))  # [batch_size, n_hidden]
        output = self.b + torch.mm(X, self.W) + torch.mm(tanh, self.U)  # [batch_size, n_class]
        return output


model = NNLM()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

input_batch, target_batch = make_batch(sentences)
input_batch = Variable(torch.LongTensor(input_batch))
target_batch = Variable(torch.LongTensor(target_batch))

# Training
for epoch in range(5000):

    optimizer.zero_grad()
    output = model(input_batch)

    # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, target_batch)
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

# Predict
predict = model(input_batch).data.max(1, keepdim=True)[1]

print(predict)
# Test
print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])