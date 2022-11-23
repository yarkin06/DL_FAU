import numpy as np
from Layers import Base

class Dropout(Base.BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.prob = probability

    def forward(self, input_tensor):
        if self.testing_phase:
            self.mask = np.ones(input_tensor.shape)
        else:
            temp = np.random.rand(*input_tensor.shape)
            self.mask = (temp < self.prob).astype(float)
            self.mask /= self.prob
        return input_tensor * self.mask
    
    def backward(self, error_tensor):
        return error_tensor * self.mask
        