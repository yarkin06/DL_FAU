import numpy as np
from Layers import Base, Helpers 
import copy

class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
        self.initialize()
        self._optimizer = None
        self.moving_mean = None
        self.moving_var = None
        self.decay = 0.8
    
    def initialize(self, weights_initializer=None, bias_initializer=None):
        self.gamma = np.ones(self.channels)
        self.beta = np.zeros(self.channels)

    def forward(self, input_tensor):
        X = input_tensor
        conv = False
        if X.ndim == 4:
            conv = True
            X = self.reformat(X)
        self.X = X
        if self.testing_phase:
            if self.moving_mean is None or self.moving_var is None:
                print("[!] BatchNormalization: You need to train the model before testing")
            self.mean = self.moving_mean
            self.var = self.moving_var
        else:
            self.mean = np.mean(X, axis= 0)
            self.var = np.var(X, axis=0)
            if self.moving_mean is None:
                self.moving_mean = copy.deepcopy(self.mean)
                self.moving_var = copy.deepcopy(self.var)
            else:
                self.moving_mean = self.moving_mean * self.decay + self.mean * (1 - self.decay)
                self.moving_var = self.moving_var * self.decay + self.var * (1 - self.decay)
        self.X_hat = (X - self.mean) / np.sqrt(self.var + np.finfo(float).eps)
        out = self.gamma * self.X_hat + self.beta
        if conv:
            out = self.reformat(out)
        return out

    def backward(self, error_tensor):
        E = error_tensor
        conv = False
        if E.ndim == 4:
            conv = True
            E = self.reformat(E)
        dgamma = np.sum(E * self.X_hat, axis=0)
        dbeta = np.sum(E, axis=0)
        grad = Helpers.compute_bn_gradients(E, self.X, self.gamma, self.mean, self.var)
        if self._optimizer is not None:
            self._optimizer.weight.calculate_update(self.gamma, dgamma)
            self._optimizer.bias.calculate_update(self.beta, dbeta)

        if conv:
            grad = self.reformat(grad)
        self.gradient_weights = dgamma
        self.gradient_bias = dbeta
        return grad
    
    def reformat(self, tensor):
        if tensor.ndim == 4:
            self.reformat_shape = tensor.shape
            B, H, M, N = tensor.shape
            tensor = tensor.reshape(B, H, M * N)
            tensor = tensor.transpose(0, 2, 1)
            tensor = tensor.reshape(B * M * N, H)
            return tensor
        else:
            B, H, M, N = self.reformat_shape
            tensor = tensor.reshape(B, M * N, H)
            tensor = tensor.transpose(0, 2, 1)
            tensor = tensor.reshape(B, H, M, N)
            return tensor

    @property
    def weights(self):
        return self.gamma

    @weights.setter
    def weights(self, gamma):
        self.gamma = gamma

    @property
    def bias(self):
        return self.beta

    @bias.setter
    def bias(self, beta):
        self.beta = beta

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weight = copy.deepcopy(optimizer)
        self._optimizer.bias = copy.deepcopy(optimizer)