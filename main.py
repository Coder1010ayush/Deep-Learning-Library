
from Tensor.tensor import visualize_graph as vz , Tensor, toNumpy,toTensor
import torch
import torch.nn
from torch import nn
from torch.nn import Linear


mat = torch.rand(10,10,requires_grad=True)
scalar = torch.rand(1,dtype=float,requires_grad=True)
scalar_next = torch.rand(1,dtype=float,requires_grad=True)
scalar_iter = torch.rand(1,dtype=float,requires_grad=True)





