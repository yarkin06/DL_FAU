import numpy as np
from Optimization import Optimizers
from Layers import Base

class SoftMax(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.lastIn = input_tensor
        temp = np.exp(input_tensor - input_tensor.max(axis = 1)[np.newaxis].T)
        self.lastOut = temp / temp.sum(axis = 1)[np.newaxis].T
        return self.lastOut

    def backward(self, error_tensor):
        return self.lastOut * (error_tensor - (error_tensor * self.lastOut).sum(axis = 1)[np.newaxis].T)