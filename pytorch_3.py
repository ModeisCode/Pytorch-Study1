import torch
import pandas as pandas
import numpy as np
import matplotlib.pyplot as plt
print(torch.__version__)

int_32_tensor = torch.tensor([3,6,9],dtype=torch.int32)
print(int_32_tensor)

float_32_tensor = torch.tensor([10,2,5],dtype=torch.float32)
result = int_32_tensor * float_32_tensor
print(result)

### getting information from tensors

some_tensor = torch.rand(3,4)
print(some_tensor.dtype)
print(some_tensor.shape)
print(some_tensor.requires_grad)
print(some_tensor.size)
print(some_tensor.device)