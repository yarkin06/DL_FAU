import numpy as np
from Layers import Base

class Sigmoid(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.activ = 1 / (1 + np.exp(-input_tensor))
        return self.activ

    def backward(self, error_tensor):
        # temp = self.forward(error_tensor)
        return self.activ * (1 - self.activ) * error_tensor