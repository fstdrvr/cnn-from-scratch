from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.mlp.fully_conn import *
from lib.mlp.layer_utils import *
from lib.cnn.layer_utils import *


""" Classes """
class TestCNN(Module):
    def __init__(self, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(3, 3, 3),
            MaxPoolingLayer(2, 2, 'maxp'),
            flatten(),
            fc(27, 5, 0.02)
            ########### END ###########
        )


class SmallConvolutionalNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(3, 3, 10, name="conv1"),
            gelu(name="gelu1"),
            dropout(0.9, name="drop1"),
            ConvLayer2D(10, 4, 10, name="conv2"),
            gelu(name="gelu2"),
            dropout(0.9, name="drop2"),
            MaxPoolingLayer(3, 3, name="maxp"),
            flatten(name="flat"),
            fc(810, 200, 0.02, name="fc1"),
            gelu(name="gelu3"),
            dropout(0.5, name="drop3"),
            fc(200, 20, 0.02, name="fc2")
            ########### END ###########
        )