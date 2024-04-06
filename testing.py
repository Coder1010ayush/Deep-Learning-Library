import torch
from Tensor.tensor import Tensor 
from Tensor.tensor import visualize_graph as vz
from activation.activation import relu,tanh,selu,gelu
import numpy as np
import os
import sys
import pathlib
import graphviz 
from graphviz import dot
import json
from torch import autograd
from torch import nn
from activation.activation import sigmoid, swish, relu, leaky_relu , selu, gelu ,tanh
from nn import linear_nn
from nn.linear_nn import Node, Module
from nn.linear_nn import Linear,flatten

def testing_addition_subtracton_auto_diff():
    # result of custom tensor class and torch auto diff must be same or very close to each-other 
    # let us start 

    #case one : with same shape 
    
    arr1 = np.array(object= [ [1.9,1.5,1.6],[12.2,9.26,6.7], [2.5,2.7,2.9] ]  )
    arr2 = np.array(object= [ [1.9,7.5,1.6],[7.2,6.16,6.7], [2.5,2.7,2.9] ]  )
    arr3 = np.array(object= [ [1.8,10.5,9.6],[11.2,6.6,6.7], [2.5,2.7,2.9] ]  )
    obj1 = Tensor(value=arr1) 
    obj2 = Tensor(value=arr2)
    obj3 = Tensor(value=arr3)

    # performing operation and calculating gradient using custom autograd and than checking with pytroch
    out1 = obj1 * obj2 
    out2 = out1 * obj3
    
    out = out2.mean()
    out.backward()
    vz(self=out,filename='testing/summaton_testing')
    with open(file='testing/summation.txt',mode="w") as file:
        file.write(f'obj1 is {obj1}\n')
        file.write(f'obj2 is {obj2}\n')
        file.write(f'obj3 is {obj3}\n')
        file.write(f'out is {out}\n')

        file.write(f'grad of obj1 is {obj1.grad}\n\n')
        file.write(f'grad of obj2 is {obj2.grad}\n\n')
        file.write(f'grad of obj3 is {obj3.grad}\n\n')
        file.write(f'grad of out1 is {out1.grad}\n\n')
        file.write(f'grad of out2 is {out2.grad}\n\n')
        file.write(f'grad of out is   {out.grad}\n\n')


    obj1 = torch.tensor(data= arr1,requires_grad=True)
    obj2 = torch.tensor(data = arr2,requires_grad=True)
    obj3 = torch.tensor(data= arr3,requires_grad=True)

    out1 = torch.matmul(obj1 , obj2)
    out1.retain_grad()
    out2 = torch.matmul(out1 , obj3)
    out2.retain_grad()
    out = out2.mean()
    out.retain_grad()
    out.backward()

    with open(file='testing/summation_torch.txt',mode="w") as file:
        file.write(f'obj1 is {obj1}\n')
        file.write(f'obj2 is {obj2}\n')
        file.write(f'obj3 is {obj3}\n')
        file.write(f'out is {out}\n')

        file.write(f'grad of obj1 is {obj1.grad}\n\n')
        file.write(f'grad of obj2 is {obj2.grad}\n\n')
        file.write(f'grad of obj3 is {obj3.grad}\n\n')
        file.write(f'grad of out1 is {out1.grad}\n\n')
        file.write(f'grad of out2 is {out2.grad}\n\n')
        file.write(f'grad of out is   {out.grad}\n\n')

    #case two : with different shape
    #case three : with different dtypes such as scalar with vector and vector with matrix and matric to matrix


def testing_sigmoid_function():
    # testing out the autograd's each functionality
    # let us start!
    # chaecking consistancy with using both scalar and tensor 
    arr1 = np.arange(start=2 , stop= 5, step= 0.5, dtype=float)
    arr2 = np.arange(start=4, stop=7, step=0.5 , dtype=float)
    arr3 = np.arange(start=2 , stop= 5, step= 0.5, dtype=float)
    arr4 = np.arange(start=4, stop=7, step=0.5 , dtype=float)
    arr5 = np.arange(start=2 , stop= 5, step= 0.5, dtype=float)
    arr6 = np.arange(start=4, stop=7, step=0.5 , dtype=float)
    
    arr7 = arr6.reshape((6,1))
    # converting all the np array into tensor class obj so that we can track the gradient flow
    obj1 = Tensor(value=arr1)
    obj2 = Tensor(value=arr2)
    obj3 = Tensor(value=arr3)
    obj4 = Tensor(value=arr4)
    obj5 = Tensor(value=arr5)
    obj6 = Tensor(value=arr6)
    obj7 = Tensor(value=arr7)
    # dfining the mathematical structure 
    out1 =  obj1 ** 5
    out2 = out1 + obj2
    out3 = sigmoid(self=out2)
    out = out3 * obj3
    out.backward()
    vz(self=out, filename="testing/auto")
    print('grad of obj1 ', obj1.grad)
    print()
    print('grad of obj2 ', obj2.grad)
    print()
    print('grad of obj3 ', obj3.grad)
    print()
    print('grad of out1 ', out1.grad)
    print()
    print('grad of out2 ', out2.grad)
    print()
    print('grad of out3 ', out3.grad)
    print()
    print('grad of out ', out.grad)
    print()
    print('out data is ' , out)

    print()
    print()
    print()
    obj1 = torch.tensor(data=arr1,requires_grad=True)
    obj2 = torch.tensor(data=arr2,requires_grad=True)
    obj3 = torch.tensor(data=arr3,requires_grad=True)
    obj4 = torch.tensor(data=arr4,requires_grad=True)
    obj5 = torch.tensor(data=arr5,requires_grad=True)
    obj6 = torch.tensor(data=arr6,requires_grad=True)
    obj7 = torch.tensor(data=arr7,requires_grad=True)

    out1 =  obj1 ** 5
    out1.retain_grad()
    out2 = out1 + obj2
    out2.retain_grad()
    out3 = out2.sigmoid()
    out3.retain_grad()
    out = torch.matmul(out3 , obj3)
    out.retain_grad()
    out.backward()

    print()
    print()
    print()
    print('grad of obj1 ', obj1.grad)
    print()
    print('grad of obj2 ', obj2.grad)
    print()
    print('grad of obj3 ', obj3.grad)
    print()
    print('grad of out1 ', out1.grad)
    print()
    print('grad of out2 ', out2.grad)
    print()
    print('grad of out3 ', out3.grad)
    print()
    print('grad of out ', out.grad)
    print()
    print('out data is ' , out)


if __name__ == '__main__':
    testing_sigmoid_function()
    testing_addition_subtracton_auto_diff()