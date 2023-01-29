import torch
import numpy as np
import math

x = torch.rand(5, 3)
print(x)

# tensor initialization from data directly
data = [[1, 2], [2, 3], [3, 4]]
x_data = torch.tensor(data)
print(x_data)

# tensor initialization from numpy array
np_array = np.array(data)
x_np_data = torch.from_numpy(np_array)
print(x_np_data)

# tensor initialization from another tensor
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones}")
x_ones_float = torch.ones_like(x_data, dtype=torch.float)
print(f"Ones Tensor float: \n {x_ones_float}")
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand}")

# tensor initialization with random values
shape = (2, 3, )
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor}")
print(f"Ones Tensor: \n {ones_tensor}")
print(f"Zeros Tensor: \n {zeros_tensor}")

# Tensor Operations

# slicing and indexing
tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)

# joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# multiplying tensors
print(f"Element wise product: tensor.mul(tensor) \n {tensor.mul(tensor)}")
print(f"Another way to do element wise product is: tensor * tensor \n {tensor*tensor}")

# in-place operations. This can save memory by can be problematic when computing
# derivatives because of list of history, hence it is discouraged.
print(f"Adding 5 to tensor: \n {tensor.add_(5)}")

# Tensor to numpy array
t = torch.ones(5)
print(f"t: \n {t}")
n = t.numpy()
print(f"n: \n {n}")

# updating tensor will also reflect in the numpy array
t.add_(1)
print(f"t: \n {t}")
print(f"n: \n {n}")

# updating numpy array can also reflect in the tensor
np.add(n, 1, out=n)
print(f"t: \n {t}")
print(f"n: \n {n}")
