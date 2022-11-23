import numpy as np
input_tensor = np.array(range(np.prod((3,10,14)) * 2), dtype=float)
print(input_tensor.shape)
input_tensor = input_tensor.reshape(2, *(3,10,14))
print(input_tensor.ndim)