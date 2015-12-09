# coding:utf-8
__author__ = 'liangz14'
from util import load2d,load
import cPickle as pickle
from lasagne import layers
from nolearn.lasagne import NeuralNet


net2 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        #('conv3', layers.Conv2DLayer),
        #('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
       # ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    #conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500,
    #hidden5_num_units=500,
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,
    max_epochs=30,
    verbose=1,
    )

X, y = load2d()  # load 2-d data

net2.fit(X, y)

with open('net2.pickle', 'wb') as f:
    pickle.dump(net2, f, -1)


