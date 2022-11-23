import numpy as np
from Layers import Base

class Pooling(Base.BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        self.lastShape = input_tensor.shape
        h_pools = np.ceil((input_tensor.shape[2] - self.pooling_shape[0] + 1) / self.stride_shape[0])
        v_pools = np.ceil((input_tensor.shape[3] - self.pooling_shape[1] + 1) / self.stride_shape[1])
        output_tensor = np.zeros((*input_tensor.shape[0:2], int(h_pools), int(v_pools)))
        self.x_s = np.zeros((*input_tensor.shape[0:2], int(h_pools), int(v_pools)), dtype=int)
        self.y_s = np.zeros((*input_tensor.shape[0:2], int(h_pools), int(v_pools)), dtype=int)
        
        a = -1
        for i in range(0, input_tensor.shape[2] - self.pooling_shape[0] + 1, self.stride_shape[0]):
            a += 1
            b = -1
            for j in range(0, input_tensor.shape[3] - self.pooling_shape[1] + 1, self.stride_shape[1]):
                b += 1
                temp = input_tensor[:, :, i:i+self.pooling_shape[0], j:j+self.pooling_shape[1]].reshape(*input_tensor.shape[0:2], -1)
                output_pos = np.argmax(temp, axis = 2)
                x = output_pos // self.pooling_shape[1]
                y = output_pos % self.pooling_shape[1]
                self.x_s[:, :, a, b] = x
                self.y_s[:, :, a, b] = y
                output_tensor[:, :, a, b] = np.choose(output_pos, np.moveaxis(temp, 2, 0))         
                #np.max(input_tensor[:, :, i:i+self.pooling_shape[0], j:j+self.pooling_shape[1]], axis =(2, 3))
                
        return output_tensor
    
    def backward(self, error_tensor):
        return_tensor = np.zeros(self.lastShape)
        for a in range(self.x_s.shape[0]):
            for b in range(self.x_s.shape[1]):
                for i in range(self.x_s.shape[2]):
                    for j in range(self.y_s.shape[3]):
                        return_tensor[a, b, i*self.stride_shape[0]+self.x_s[a, b, i, j], j*self.stride_shape[1]+self.y_s[a, b, i, j]] += error_tensor[a, b, i, j]
        return return_tensor