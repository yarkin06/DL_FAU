import copy
import numpy as np

class NeuralNetwork(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = list()
        self.layers = list()
        self.data_layer = None
        self.loss_layer = None
    
    def forward(self):
        data, self.label = copy.deepcopy(self.data_layer.next())
        for layer in self.layers:
            data = layer.forward(data)
        return self.loss_layer.forward(data, copy.deepcopy(self.label))

    def backward(self):
        y = copy.deepcopy(self.label)
        y = self.loss_layer.backward(y)
        for layer in reversed(self.layers):
            y = layer.backward(y)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)
    
    def train(self, iterations):
        for epoch in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor