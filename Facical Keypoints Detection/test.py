# coding:utf-8
__author__ = 'liangz14'
import cPickle as pickle
from util import load2d,load
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np
from pandas.io.parsers import read_csv
import datetime

import os
FLOOKUP = 'IdLookupTable.csv'
def plot_sample(x, y, axis,y_true):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, c='b',marker='x', s=10)
    axis.scatter(y_true[0::2] * 48 + 48, y_true[1::2] * 48 + 48,c='r', marker='x', s=10)
'''
#f1 = open('net1.pickle', 'r')
f2 = open('net2.pickle', 'r')

#net1 = pickle.load(f1)
net2 = pickle.load(f2)

#X1,y1 = load()
X2 = load2d(test=True)

sample1,y_true1 = X1[6:7],y1[6]
sample2,y_true2 = X2[6:7],y2[6]

y_pred1 = net1.predict(sample1)[0]
y_pred2 = net2.predict(sample2)[0]

fig = plt.figure(figsize=(6, 3))
ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])

plot_sample(sample1[0], y_pred1, ax,y_true1)

ax = fig.add_subplot(2, 2, 2, xticks=[], yticks=[])

plot_sample(sample2[0], y_pred2, ax,y_true2)

plt.show()
'''


def predict(fname_specialists='net2.pickle'):
    with open(fname_specialists, 'rb') as f:
        net = pickle.load(f)

        X = load2d(test=True)[0]

        y_pred = net.predict(X)

        y_pred2 = y_pred * 48 + 48
        y_pred2 = y_pred2.clip(0, 96)

        df = DataFrame(y_pred2)

        lookup_table = read_csv(os.path.expanduser(FLOOKUP))
        values = []

        for index, row in lookup_table.iterrows():
            values.append((
                row.RowId,
                y_pred2[int(row.ImageId)-1][int(row.RowId)%30-1]
                ))



        submission = DataFrame(values, columns=('RowId','Location'))
        filename = 'submission1.csv'

        submission.to_csv(filename, index=False)
        print("Wrote {}".format(filename))

predict(fname_specialists='net2.pickle')