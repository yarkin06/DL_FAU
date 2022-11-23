import numpy as np
from Optimization import Optimizers
from Layers import Base
import copy

class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(size = (input_size, output_size))
        self.bias = np.random.uniform(size = (1, output_size))
        self.gradient_weights = None
        self.gradient_bias = None
        self._optimizer = None
        self.temp = []

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weight = copy.deepcopy(optimizer)
        self._optimizer.bias = copy.deepcopy(optimizer)

    def forward(self, input_tensor):
        self.lastIn = input_tensor
        self.lastOut = np.dot(input_tensor, self.weights) + self.bias
        return self.lastOut
     
    def backward(self, error_tensor):
        dx = np.dot(error_tensor, self.weights.T)
        dW = np.dot(self.lastIn.T, error_tensor)
        db = np.sum(error_tensor, axis = 0)
        if self._optimizer != None:
            self.weights = self._optimizer.weight.calculate_update(self.weights, dW)
            self.bias = self._optimizer.bias.calculate_update(self.bias, db)
       
        self.gradient_bias = error_tensor
        self.gradient_weights = dW
       
        return dx

    #From HW3 append:

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, self.input_size, self.output_size)
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.output_size)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights
    
    
