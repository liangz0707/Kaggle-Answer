# coding:utf-8
__author__ = 'liangz14'
# file kfkd.py
import os
import matplotlib.pyplot  as pyplot
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import cPickle as pickle
from util import load2d,load

from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 9216),  # 96x96 input pixels per batch
    hidden_num_units=100,  # number of units in hidden layer
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=30,  # 30 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=400,  # we want to train this many epochs
    verbose=1,
    )

X, y = load()

net1.fit(X, y)

with open('net1.pickle', 'wb') as f:
    pickle.dump(net1, f, -1)

