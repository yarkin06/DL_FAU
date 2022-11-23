import numpy as np
import copy
from Layers import Base
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH

class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self._memorize = False
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.FC_h = FullyConnected(hidden_size + input_size, hidden_size)
        self.FC_y = FullyConnected(hidden_size, output_size)
        self.gradient_weights_n = np.zeros((self.hidden_size + self.input_size + 1, self.hidden_size))
        self.weights_y = None
        self.weights_h = None
        self.weights = self.FC_h.weights
        self.tan_h = TanH()
        self.bptt = 0
        self.h_t = None
        self.prev_h_t = None
        self.batch_size = None
        self.optimizer = None
        self.h_mem = []

    def forward(self, input_tensor):
        self.batch_size = input_tensor.shape[0]
        if self._memorize:
            if self.h_t is None:
                self.h_t = np.zeros((self.batch_size + 1, self.hidden_size))
            else:
                self.h_t[0] = self.prev_h_t
        else:
            self.h_t = np.zeros((self.batch_size + 1, self.hidden_size))

        y_t = np.zeros((self.batch_size, self.output_size))

        # concatenating x,ht-1 and 1 to do forwarding to obtain new hidden state ht
        # 1: for t from 1 to T do:
        # 2:    ut = W hh · h t − 1 + W xh · x t + b h --> h t = tanh (x̃ t · W h )
        # 3:    h t = tanh ( u t )
        # 4:    o t = W hy · h t + b y
        # 5:    ŷ t = σ( o t )

        for b in range(self.batch_size):
            hidden_ax = self.h_t[b][np.newaxis, :]
            input_ax = input_tensor[b][np.newaxis, :]
            # x̃_t:
            input_new = np.concatenate((hidden_ax, input_ax), axis = 1)

            self.h_mem.append(input_new)

            w_t = self.FC_h.forward(input_new)
            input_new = np.concatenate((np.expand_dims(self.h_t[b], 0), np.expand_dims(input_tensor[b], 0)), axis=1)
            self.h_t[b+1] = TanH().forward(w_t) # h t = tanh (x̃ t · W h )
            y_t[b] = (self.FC_y.forward(self.h_t[b + 1][np.newaxis, :]))
        
        self.prev_h_t = self.h_t[-1]
        self.input_tensor = input_tensor

        return y_t

    def backward(self, error_tensor):

        self.out_error = np.zeros((self.batch_size, self.input_size))

        self.gradient_weights_y = np.zeros((self.hidden_size + 1, self.output_size))
        self.gradient_weights_h = np.zeros((self.hidden_size+self.input_size+1, self.hidden_size))

        count = 0

        grad_tanh = 1-self.h_t[1::] ** 2
        hidden_error = np.zeros((1, self.hidden_size))

        # 1: for t from 1 to T do:
        # 2:    Run RNN for one step, computing h_t and y_t
        # 3:    if t mod k_1 == 0:
        # 4:        Run BPTT from t down to t-k_2
        
        for b in reversed(range(self.batch_size)):
            yh_error = self.FC_y.backward(error_tensor[b][np.newaxis, :])
            self.FC_y.input_tensor = np.hstack((self.h_t[b+1], 1))[np.newaxis, :]

            grad_yh = hidden_error + yh_error
            grad_hidden = grad_tanh[b]*grad_yh
            xh_error = self.FC_h.backward(grad_hidden)
            hidden_error = xh_error[:, 0:self.hidden_size]
            x_error = xh_error[:, self.hidden_size:(self.hidden_size + self.input_size + 1)]
            self.out_error[b] = x_error

            con = np.hstack((self.h_t[b], self.input_tensor[b],1))
            self.FC_h.input_tensor = con[np.newaxis, :]
            # print(count, "", self.bptt)
            if count <= self.bptt:
                self.weights_y = self.FC_y.weights
                self.weights_h = self.FC_h.weights
                self.gradient_weights_y = self.FC_y.gradient_weights
                self.gradient_weights_h = self.FC_h.gradient_weights
            count += 1

        if self.optimizer is not None:
            self.weights_y = self.optimizer.calculate_update(self.weights_y, self.gradient_weights_y)
            self.weights_h = self.optimizer.calculate_update(self.weights_h, self.gradient_weights_h)
            self.FC_y.weights = self.weights_y
            self.FC_h.weights = self.weights_h
        return self.out_error

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = copy.deepcopy(optimizer)

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    def initialize(self, weights_initializer, bias_initializer):
        self.FC_y.initialize(weights_initializer, bias_initializer)
        self.FC_h.initialize(weights_initializer, bias_initializer)

    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def gradient_weights(self):
        return self.gradient_weights_n

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.FC_y.gradient_weights = gradient_weights