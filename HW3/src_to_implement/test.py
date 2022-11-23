import numpy as np
from Layers import Pooling as P

stride_shape = (2, 2)
pooling_shape = (2, 2)

layer = P.Pooling(stride_shape, pooling_shape)
input_tensor = [[[1,3,5,3], [8,5,7,6], [5,1,6,8], [10,9,5,1]], [[1,2,3,4], [5,6,7,8], [9,10,9,8], [7,6,5,4]]]
input_tensor = np.array(input_tensor)
input_tensor = input_tensor[np.newaxis, :, :, :]

print(layer.forward(input_tensor))
error_tensor = [[[1,2],[3,4]], [[5,6],[7,8]]]
error_tensor = np.array(error_tensor)
error_tensor = error_tensor[np.newaxis, :, :, :]
for i in range(layer.x_s.shape[0]):
    for j in range(layer.x_s.shape[1]):
        print(layer.x_s[i, j, :, :])
# print(layer.backward(error_tensor))