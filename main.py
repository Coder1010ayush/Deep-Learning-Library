import torch
from torch.autograd import Variable
import torch.onnx
import torchvision.models as models
import numpy as np

from Tensor.matrix import Tensor
from nn.linear import NeuralNode,Layers,Dense
if __name__ == '__main__':
    #making a single neurons
    # obj = NeuralNode(number_of_nodes=10,act=True)

    # print(obj)
    # print(obj.parameters())
    # print(len(obj.parameters()))

    #making a layer of node 
    # layer1 = Layers(number_of_input_nodes=10,number_of_output_nodes=10,act=True)
    # print(layer1)
    # print(layer1.parameters())
    # print(len(layer1.parameters()))


    #making a dense layer of a list of layers
    # dense_obj = Dense(number_of_input=10,list_of_layers=[10,10,1],act=True)
    # print(dense_obj)
    # print()
    # print(dense_obj.parameters())
    # print(len(dense_obj.parameters()))

    # matrix = nDTensor()
    # obj = matrix.normalized_tensor(size=(4,4))

    # matrix1 = nDTensor()
    # obj = matrix1.normalized_tensor(size=(4,4))
    # # print(matrix*matrix1)

    # nd_tensor1 = Tensor(value=[[1,2],[3,4]])
                       
    # nd_tensor2 = Tensor(value=[[5,6],[7,8]])


    # print(matrix.multiply(mat1=nd_tensor1,mat2=nd_tensor2))
    

    # let us check out that backward graduent calculation working fine or not
    # a = Tensor(value=[1,2,3])
    # b = Tensor(value=[3,4,5])

    # c = 2*a + b
    # print(c)
    # print()
    # c.backward()
    # print(c)
    # print(a) 

    # let us work with torch autograd to know how gradient and backpropogation works in matrix

    # x = torch.tensor(data=[1,2,3],requires_grad=True,dtype=float)
    # y = torch.tensor(data=[1,2,3],requires_grad=True,dtype=float)

    # a = torch.tensor(data=[1,2,3],requires_grad=True,dtype=float)
    # b = torch.tensor(data=[1,2,3],requires_grad=True,dtype=float)

    # p = x + y
    # q = a*b

    # z = p*q
    # z.backward(torch.ones_like(input=z))
    # print("p matrix is : ",p)
    # print()
    # print("q matrix is : ",q)
    # print()
    # print("grad of a : ",a.grad)
    # print()
    # print("grad of b : ",b.grad)
    # print()
    # print("grad of x : ",x.grad)
    # print()
    # print("grad of y : ",y.grad)

    # # print()
    # # print()
    # # print(torch.matmul(input=x+y,other=b.T))

    # # print()
    # # print()
    # # print(torch.matmul(input=p,other=a.T))

