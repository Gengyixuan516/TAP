import numpy as np
import sparse
from tensorly.contrib.sparse import tensor as sparse_tensor
import tensorly as tl

tl.set_backend('numpy')

indices = np.array([[0, 1, 1, 2, 2, 0], [1, 2, 0, 0, 1, 2], [2, 0, 1, 1, 2, 0]])   # 这里要求必须是整数
values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
shape = (3, 3, 3)
sparse_torch = sparse.COO((values, indices), shape=shape)

# TensorLy COO稀疏张量封装
tensor = sparse_tensor(sparse_torch)

# 分解
core, factors = tl.decomposition.tucker(tensor, rank=(1,1,1))

import torch
from tensorly.contrib.sparse import tensor as sparse_tensor

coords = torch.tensor([[1, 3],  # x 坐标
                       [2, 0],  # y 坐标
                       [3, 1]]) # z 坐标
data = torch.tensor([5, 8], dtype=torch.float32)
shape = (4, 4, 4)

# PyTorch
sparse_torch = torch.sparse_coo_tensor(coords, data, size=shape)

# TensorLy
sparse_tl = sparse_tensor(sparse_torch)

import torch
from tensorly.contrib.sparse import tensor as sparse_tensor
from tensorly.contrib.sparse.decomposition import parafac as sparse_parafac

# import tensorly.contrib.sparse as stl
# import sparse
# from tensorly.contrib.sparse.cp_tensor import cp_to_tensor
# import tensorly as tl
# tl.set_backend('numpy')
# # 1. 构造一个高稀疏张量
# shape = (1000, 1001, 1002)
# rank = 5
#
# starting_weights = stl.ones((rank))
# print(starting_weights)
#
# starting_factors = [sparse.random((i, rank)) for i in shape]
# print(starting_factors)
#
# tensor = cp_to_tensor((starting_weights, starting_factors))
# print(tensor)

# tl.set_backend('numpy') # 这里会报错，COO格式的张量不支持PyTorch的稀疏张量，所以只能用numpy或者转为dense的张量
# # 2. 使用TensorLy的密集CP分解
# from tensorly.decomposition import parafac # The dense version
# import time
# t = time.time()
# dense_cp = parafac(tensor, 5, init='random')
# print(time.time() - t)
#
# # The sparse version
# from tensorly.contrib.sparse.decomposition import parafac as sparse_parafac
# t = time.time()
# sparse_cp = sparse_parafac(tensor, 5, init='random')
# print(time.time() - t)

