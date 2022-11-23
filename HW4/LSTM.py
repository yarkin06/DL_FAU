import numpy as np
from Layers import Base, FullyConnected, TanH, Sigmoid

class LSTM(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):#13, 7, 4
        super().__init__()
        self.trainable = True
        self.memorize = False
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros((1, hidden_size))
        self.cell_state = np.zeros((1, hidden_size))
        self.FC_h = FullyConnected.FullyConnected(input_size + hidden_size, 4*hidden_size)
        self.FC_n = FullyConnected.FullyConnected(4*hidden_size, input_size + hidden_size)
        self.FC_y = FullyConnected.FullyConnected(hidden_size, output_size)
        self.weights = self.FC_h.weights

    def initialize(self, weights_initializer, bias_initializer):
        self.FC_h.initialize(weights_initializer, bias_initializer)
        self.FC_y.initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor):
        x = input_tensor
        self.hidden_state = np.zeros((input_tensor.shape[0], self.hidden_size))
        self.cell_state = np.zeros((input_tensor.shape[0], self.FC_h.output_size))
        x = np.concatenate((x, self.hidden_state), axis=1)
        x1 = x.copy()
        FC_x = self.FC_h.forward(x1)
        Cell_line = FC_x * self.cell_state
        Tan_FC_x = self.FC_h.forward(FC_x)
        Mid = Tan_FC_x * FC_x
        Cell_line = Cell_line + Mid
        del Mid
        self.cell_state = Cell_line
        Cell_line = TanH.TanH().forward(Cell_line)
        self.hidden_state = FC_x * Cell_line
        del FC_x
        return self.FC_y.forward(self.hidden_state)
        
    def backward(self, grad_output):
        pass

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value