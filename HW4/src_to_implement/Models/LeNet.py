import numpy as np
import NeuralNetwork as nn
from Layers import *
from Optimization import *

def build():
    optimizer = Optimizers.Adam(5e-4, 0.9, 0.999)
    optimizer.add_regularizer(Constraints.L2_Regularizer(4e-4))
    net = nn.NeuralNetwork(optimizer, Initializers.He(), Initializers.He())
    net.append_layer(Conv.Conv((1, 1), (6, 5, 5), 6))
    net.append_layer(ReLU.ReLU())
    net.append_layer(Pooling.Pooling((2, 2), (2, 2)))
    net.append_layer(Conv.Conv((1, 1), (6, 5, 5), 16))
    net.append_layer(ReLU.ReLU())
    net.append_layer(Pooling.Pooling((2, 2), (2, 2)))
    net.append_layer(Flatten.Flatten())
    net.append_layer(FullyConnected.FullyConnected(16 * 7 * 7, 120))
    net.append_layer(ReLU.ReLU())
    net.append_layer(FullyConnected.FullyConnected(120, 84))
    net.append_layer(ReLU.ReLU())
    net.append_layer(FullyConnected.FullyConnected(84, 10))
    net.append_layer(SoftMax.SoftMax())
    net.loss_layer = Loss.CrossEntropyLoss()
    return net
