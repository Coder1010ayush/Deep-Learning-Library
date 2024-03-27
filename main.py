from matplotlib import pyplot as plt
import torch
from torch.autograd import Variable
import torch.onnx
import torchvision.models as models
import numpy as np
from sklearn.datasets import make_moons, make_blobs
from Tensor.matrix import Tensor
from nn.linear import NeuralNode,Layers,Dense
from sklearn import model_selection
from sklearn import metrics
from torch import nn
from torch import functional as F
from torch import _torch_docs
from nn.cnn import conv
from nn.pooling_single_channel import MaxPool
if __name__ == '__main__':
#     # making a single neurons
#     # obj = NeuralNode(number_of_nodes=10,act=True)

#     # print(obj)
#     # print(obj.parameters())
#     # print(len(obj.parameters()))

#     #making a layer of node 
#     # layer1 = Layers(number_of_input_nodes=10,number_of_output_nodes=10,act=True)
#     # print(layer1)
#     # print(layer1.parameters())
#     # print(len(layer1.parameters()))


#     #making a dense layer of a list of layers
#     # dense_obj = Dense(number_of_input=10,list_of_layers=[10,10,1],act=True)
#     # print(dense_obj)
#     # print()
#     # print(dense_obj.parameters())
#     # print(len(dense_obj.parameters()))

#     # matrix = nDTensor()
#     # obj = matrix.normalized_tensor(size=(4,4))

#     # matrix1 = nDTensor()
#     # obj = matrix1.normalized_tensor(size=(4,4))
#     # # print(matrix*matrix1)

#     # nd_tensor1 = Tensor(value=[[1,2],[3,4]])
                       
#     # nd_tensor2 = Tensor(value=[[5,6],[7,8]])


#     # print(matrix.multiply(mat1=nd_tensor1,mat2=nd_tensor2))
    

#     # let us check out that backward graduent calculation working fine or not
#     # a = Tensor(value=[1,2,3])
#     # b = Tensor(value=[3,4,5])

#     # c = 2*a + b
#     # print(c)
#     # print()
#     # c.backward()
#     # print(c)
#     # print(a) 

#     # let us work with torch autograd to know how gradient and backpropogation works in matrix

#     # x = torch.tensor(data=[1,2,3],requires_grad=True,dtype=float)
#     # y = torch.tensor(data=[1,2,3],requires_grad=True,dtype=float)

#     # a = torch.tensor(data=[1,2,3],requires_grad=True,dtype=float)
#     # b = torch.tensor(data=[1,2,3],requires_grad=True,dtype=float)

#     # p = x + y
#     # q = a*b

#     # z = p*q
#     # outcome = z.sigmoid()
#     # outcome.backward(torch.ones_like(input=z))
#     # print("grad of a : ",a.grad)
#     # print("grad of b : ",b.grad)
#     # print("grad of x : ",x.grad)
#     # print("grad of y : ",y.grad)

#     # x = Tensor(value=[1,2,3])
#     # y = Tensor(value=[1,2,3])

#     # a = Tensor(value=[1,2,3])
#     # b = Tensor(value=[1,2,3])

#     # p = x + y
#     # q = a * b

#     # z = p*q
#     # outcome = z.sigmoid()
#     # outcome.backward()
#     # print()
#     # print()
#     # print("grad of x : ",x.grad)
#     # print("grad of y : ",y.grad)
#     # print("grad of a : ",a.grad)
#     # print("grad of b : ",b.grad)



#     # testing out a dense layer neural network

#     # nn = Dense(number_of_input=2,list_of_layers=[16,16,1],act=True)
#     # print()
#     # print(nn)
#     # print("number of parameters : ", len(nn.parameters()))
#     # X, y = make_moons(n_samples=100, noise=0.1)
#     # y = y*2 - 1
#     # plt.figure(figsize=(5,5))
#     # plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='jet')
#     # plt.show()


#     # print()
#     # print()

#     # start working with torch conv class
#     # conv1 = torch.nn.Conv1d(in_channels=2,out_channels=3,kernel_size=3,bias=False)
#     # x = torch.rand(size=(10,2,10))
#     # print("data is : ", x)
#     # print()
#     # print("conv1d is : ",conv1)
#     # print()
#     # out = conv1(x)
#     # print(out)
#     # print("shape of out : ",out.shape)


#     # testing cnn conv layer class

#     # conv_layer = conv(image_shape=(3,10,10),kernal_size=3,number_of_layers_of_kernal=5)
#     # x = Tensor(value=np.random.random(size=(100,10,10)))
#     # output = conv_layer.convolve(x)
#     # print('ouptput is ' , output)
#     # print('shape of output is ', output.shape)

#     # print()
#     # print()
    
#     # # backpropogate from here
#     # grad = Tensor(value=np.random.random(size=output.shape))
#     # grad = conv_layer.backward(output_gradient=grad.data,learning_rate=0.001)
#     # print(grad)


#     # start working on pooling layer
#     # arr = np.random.randint(low=10,high=1000,size=(7,7))
#     # print('data is ',arr)
#     # kernal_size = 3
#     # stride = 2
#     # row , col = arr.shape
#     # cnt = 0
#     # for i in range(row):
#     #     if cnt ==3:
#     #         break

#     #     if i+kernal_size > row:
#     #         break
#     #     for j in range(col):
#     #         if j+kernal_size > col:
#     #             break
#     #         mat = arr[i : i+kernal_size,j : j+kernal_size]
#     #         print(mat)

#     #         j += stride
#     #     i += stride
#     #     cnt += 0

#     # print(arr)
#     # kernal_size = 3 
#     # for batch,data in enumerate(arr):
#     #     print('batch is : ',batch)
#     #     row, col = data.shape
#     #     for i in range(0,row):
#     #         print(i)

#     # arr = np.random.randint(low=10,high=1000,size=(7,7))
#     # print(arr)
#     # print(np.max(a=arr))
#     arr = torch.randint(low=10,high=500,size=(10,32,31))
#     pool1 = nn.MaxPool2d(kernel_size=6,stride=2,padding=3)
#     pool2 = MaxPool(kernel_size=6,stride=2,padding=3)
#     y1 = pool1(arr)
#     print(y1.shape)
#     arr = Tensor(value=arr)
#     y2 = pool2.maxpool(arr)
#     print(y2.shape())
      pass


    