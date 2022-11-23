import numpy as np
from Optimization import Optimizers
from Layers import Base

class ReLU(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.lastIn = input_tensor
        self.lastOut = np.maximum(0, input_tensor) 
        return self.lastOut

    def backward(self, error_tensor):
        return np.where(self.lastIn > 0, error_tensor, 0)