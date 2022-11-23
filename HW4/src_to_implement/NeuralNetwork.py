import copy
import numpy as np
import pickle
import time 

def save(filename, net):
    pickle.dump(net, open(filename, 'wb'))

def load(filename, data_layer):
    net = pickle.load(open(filename, 'rb'))
    net.data_layer = data_layer
    return net

class NeuralNetwork(object):
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self._phase = None
        self.loss = list()
        self.layers = list()
        self.data_layer = None
        self.loss_layer = None

        self.weights_initializer = copy.deepcopy(weights_initializer)
        self.bias_initializer = copy.deepcopy(bias_initializer)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['data_layer']
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase):
        self._phase = phase

    def forward(self):
        data, self.label = copy.deepcopy(self.data_layer.next())
        reg_loss = 0
        for layer in self.layers:
            layer.testing_phase = False
            data = layer.forward(data)
            if self.optimizer.regularizer is not None:
                reg_loss += self.optimizer.regularizer.norm(layer.weights)
        glob_loss = self.loss_layer.forward(data, copy.deepcopy(self.label))
        return glob_loss + reg_loss

    def backward(self):
        y = copy.deepcopy(self.label)
        y = self.loss_layer.backward(y)
        for layer in reversed(self.layers):
            y = layer.backward(y)

    def append_layer(self, layer):
        if layer.trainable:
            layer.initialize(self.weights_initializer, self.bias_initializer)
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)
    
    def train(self, iterations):
        self.phase = 'train'
        for epoch in range(iterations):
            start = time.time()
            # print('Epoch: %4d'%(epoch+1), end = ' ')
            loss = self.forward()
            self.loss.append(loss)
            self.backward()
            stop = time.time()
            # print('%.2f'%(stop-start))

    def test(self, input_tensor):
        self.phase = 'test'
        for layer in self.layers:
            layer.testing_phase = True
            input_tensor = layer.forward(input_tensor)
        return input_tensor