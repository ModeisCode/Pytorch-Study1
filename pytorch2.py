import torch
import pandas as pandas
import numpy as np
import matplotlib.pyplot as plt
print(torch.__version__)

#scalar
scalar = torch.tensor(7)

print(scalar.ndim)

#Get tensor back as int
print(scalar.item())

#Vector
vector = torch.tensor([7,7])

print(vector.shape)

# MATRIX
MATRIX = torch.tensor([[7,8],
                      [10,5]])

print(MATRIX)
print(f"MATRIX shape is {0}" ,MATRIX.shape)

print("----------")
# TENSOR
TENSOR = torch.tensor([[[1,2,3],
                        [3,6,9],
                        [2,4,5]]])

print(TENSOR.shape)
print(TENSOR.ndim)

# Random tensors
random_tensor = torch.rand(1,10,10)
print(random_tensor)
print(random_tensor.ndim)

# random tensor with similar shape to an image tensor
random_image_size_tensor = torch.rand(size=(3,224,224)) # height width colur channels RGB
print(random_image_size_tensor.shape)
print(random_image_size_tensor.ndim)
print(random_image_size_tensor)

### Zeros and ones
zeros = torch.zeros(3,4)
ones = torch.ones(3,4)
print(zeros)
print(ones)











