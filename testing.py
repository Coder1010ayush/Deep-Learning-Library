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
    # setting up a linear layer from skretch

    # let us assume we a data of 2 columns and 10 rows in which we have 1 column is a input vector feature and 1 column vector is a outcome vector
    # weight and bias will be defined 

    # x = Tensor(value=np.random.random(size=(10 , 1)))
    # y = Tensor(value=np.random.random(size=(10)))
    # weights = Tensor(value=np.random.random(size=(1,1)))
    # bias = Tensor(value=np.random.random(size=(1,1))[0][0])
    
    # out1 = x*  weights
    # out = out1 + bias
    # final_out = out.mean()
    # print(final_out)
    # final_out.backward()
    # vz(self=final_out, filename="testing/linear")

    # weights.data -= 0.1 * weights.grad
    # bias.data -= 0.1 * bias.grad
    # out1 = x*  weights
    # out = out1 + bias
    # final_out = out.mean()
    # print(final_out)
    # final_out.backward()

    # weights.data -= 0.1 * weights.grad
    # bias.data -= 0.1 * bias.grad
    # out1 = x*  weights
    # out = out1 + bias
    # final_out = out.mean()
    # print(final_out)
    # final_out.backward()
    # testing_addition_subtracton_auto_diff()

    # node = Node(n_in=10)
    # print(node)
    # # for params in node.parameters():
    # #     print(params)
    # #     print(type(params))

    # data = Tensor(value=np.random.random(size=(100,10)))
    # out = node(x=data)
    # print(out.data.shape)
    # print(out)
    # out.backward()

    # print(node.info())

    layer = Linear(in_feature=10, out_feature=3)
    # print('layer is ', layer)
    # for params in layer.parameters():
    #     print(params)
    #     print(params.data.shape)
    #     print()

    # print()
    # data = Tensor(value=np.random.random(size=(50,100,20,10)))
    # out = layer(data)
    # print(out.data.shape)
    # out.backward()
    # print('rendered!')
    model = Linear(in_feature=1, out_feature=1)

    # Generate some random data
    X = np.random.randn(10,20, 1)
    X.dtype = float
    x = X.reshape((200,1))
    y = 3 * x + 2 + np.random.random(size=(200,1)) * 0.1  # Linear relationship with some noise
    y.dtype = float
    # Convert data to custom tensor class
    X_tensor = Tensor(value=X)
    y_tensor = Tensor(value=y)
    print(X_tensor.data.shape)
    print(y_tensor.data.shape)
    # Training loop
    learning_rate = 0.01
    for epoch in range(70):
        # Forward pass
        predictions,final_out = model(X_tensor)
        print(final_out.data.shape)
        loss = model.mse_loss(predictions, y_tensor)
        
        # Backward pass
        model.backward(loss)
        # Update parameters
        model.update_parameters(learning_rate=0.01)
        
        # Print loss every 100 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.data}')

    