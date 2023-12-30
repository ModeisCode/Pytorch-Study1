import torch
import pandas as pandas
import numpy as np
import matplotlib.pyplot as plt
print(torch.__version__)

# range
zero_to_thousand = torch.arange(start=0,end=1000,step=77),
one_to_ten = torch.arange(start=0,end=10,step=1)
print(one_to_ten)
print(zero_to_thousand)

# tensors
ten_zeros = torch.zeros_like(input=one_to_ten)
matrix_like = torch.zeros_like(input=torch.rand(2,2))
matrix_one_like = torch.ones_like(input=torch.rand(1,3,3))
print(matrix_one_like)
print(matrix_like)
print(ten_zeros)

# Float 32 tensor
float_32_tensor = torch.tensor([3.0,6.0,9.0],
                               dtype=None, # data type float32,float16,double,bfloat -> precision
                               device=None, # what device is your tensor on ("cpu" or "cuda") 
                               requires_grad=False) # track gradients

print(float_32_tensor)
