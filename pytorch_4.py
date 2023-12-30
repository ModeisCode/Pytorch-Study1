import torch
import pandas as pandas
import numpy as np
import matplotlib.pyplot as plt

# tensor
tensor = torch.tensor([1,2,3])
print(tensor + 10)
print(tensor * 10)
print(tensor / 10)
print(tensor - 10)
print(torch.mul(tensor,10))
print(torch.add(tensor,10))

# MATRIX MULTIPLICATION
'''
Element-wise multiplication
Matrix multiplication (dot product)
dot product symbol
'''

# Element-wise multiplication
print(tensor * tensor)

# Matrix multiplication
print(torch.matmul(tensor,tensor))
print(1*1)

value = 0
for i in range(len(tensor)):
    value += tensor[i] * tensor[i]
print(value)

# torch matmul is faster for matrix multiplication
torch.matmul(tensor,tensor)

'''
1.**Inner dimensions** must match
(3,2) @ (3,2) this will not work
(2,3) @ (3,2) this will work because 3 == 3
(3,2) @ (2,3) will work because 2 == 2
The resulting matrix has the shape of the **outer dimensions**
* (2,3) @ (3,2) -> (2,2)
'''

print(tensor @ tensor) # matrix multiplication with @ symbol

# shapes for matrix multiplcation

tensor_A = torch.tensor([[1,2],
                        [3,4],
                        [5,6]])

tensor_B = torch.tensor([[7,10],
                        [8,11]])

result = torch.mm(tensor_A,tensor_B) # mm is matmul alias
print(result)

print(tensor_A.T)