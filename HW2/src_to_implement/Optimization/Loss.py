import numpy as np

class CrossEntropyLoss(object):
    
    def __init__(self):
        pass
        
    def __call__(self, *args):
        return self.forward(*args)
        
    def forward(self, prediction_tensor, label_tensor):
        self.lastIn = prediction_tensor
        y_hat = prediction_tensor
        y = label_tensor
        loss = -np.sum(y * np.log(y_hat + np.finfo(float).eps))
        return loss
    
    def backward(self, label_tensor):
        return -(label_tensor / (self.lastIn + np.finfo(float).eps))