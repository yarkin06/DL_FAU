import numpy as np
from Optimization import Optimizers
from Layers import Base

class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
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

    def forward(self, input_tensor):
        self.lastIn = input_tensor
        self.lastOut = np.dot(input_tensor, self.weights) + self.bias
        return self.lastOut
     
    def backward(self, error_tensor):
        dx = np.dot(error_tensor, self.weights.T)
        dW = np.dot(self.lastIn.T, error_tensor)
        if self._optimizer != None:
            self.weights = self._optimizer.calculate_update(self.weights, dW)
            self.bias = self._optimizer.calculate_update(self.bias, error_tensor)
       
        self.gradient_bias = error_tensor
        self.gradient_weights = dW
       
        return dx
    
    
