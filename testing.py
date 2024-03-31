import os
import sys
import numpy as np
import json
from Tensor.matrix import Tensor
from nn.cnn import conv
from nn.linear import Node , Layer , Dense
from nn.pooling_single_channel import MaxPool, MinPool , AveragePool
from nn.pooling_multi_channel import MaxPool3D,AveragePool3D,MinPool3D


def testing_autograd_basic_scalar_expr():

    # expression is out_come = x + y - z + p*q
    # building above expression using Tensor class 
    # let us start!
    x = Tensor(value=10)
    y = Tensor(value=20)
    z = Tensor(value=5)
    p = Tensor(value=6)
    q = Tensor(value=10)

    intermediate_x = x + y
    intermediate_z = intermediate_x - z
    second = p * q
    outcome = intermediate_z + second

    outcome.backward()
    outcome.visualize_graph(filename = "example1")
    print("hello")


def testing_autograd_basic_matrix_operation():
    # expression is out_come = x + y - z + p*q
    # building above expression using Tensor class 
    # let us start!
    x = Tensor(value=np.random.randint(low=10, high=100,size=(10,10)))
    y = Tensor(value=np.random.randint(low=1,high=20 , size=(10,10)))
    z = Tensor(value=np.random.randint(low=10 , high=50, size=(10 , 10)))
    intemediate_val = x + y
    intemediate_val1  = intemediate_val - z

    p = Tensor(value=np.random.randint(low=10, high=100,size=(10,10)))
    q = Tensor(value=np.random.randint(low=1,high=20 , size=(10,10)))
    out = p*q
    outcome = intemediate_val1 + out
    outcome.backward()
    outcome.visualize_graph(filename = 'example2')

def test_linear_layers():
    """
        testing linear layers from nn.linear
    """
    layer = Layer(n_input=10 , n_out= 20)
    print(layer)
    print(layer.parameters())
    print()
    print('parameters are : ', layer.parameters())
    print()
    assert(len(layer.parameters()) == 220)
    print('number of trainable parameters are : ',len(layer.parameters()))


def test_node():
    """"
        testing single neuron from nn.linear 
    
    """
    node = Node(n_input=10)
    print('node is : ',node)
    print('parameters are : ', node.parameters())

    node = Node(n_input=6)
    print('node is : ',node)
    


def test_conv_layer():
    """
        testing out conv layer working similar as torch.nn.conv
    """
    image_shape = (10,32,32)
    kernal_size = 3
    number_of_kernals = 3
    instance_conv_layer = conv(image_shape=image_shape, kernal_size=kernal_size , number_of_layers_of_kernal=number_of_kernals)
    image_data = Tensor(value= np.random.random(size=image_shape))
    out = instance_conv_layer.convolve(image_data)
    grad = instance_conv_layer.backward(output_gradient= out, learning_rate= 0.001)
    print('output data is : ',out)
    print('shape of output data is : ', out.shape)
    print()
    print()
    print('gradient calculated through backpropogation is : ',grad)
    print('length of gradient matrix is ', grad.shape)

def test_dense_layer():
    """
        testing dense layer from nn.linear 
    """
    dense_layer = Dense(number_of_input=10 , list_of_layers=[10,10,1])
    print(dense_layer)
    print(dense_layer.parameters())
    print(len(dense_layer.parameters()))

def test_maxpool_layer():
    """
        testing out max pooling layer 
    
    """
    maxpool_layer = MaxPool(kernel_size=3, stride=2,padding=1)
    image_data = Tensor(value=np.random.random(size=(10,32,32)))
    out = maxpool_layer.maxpool(x=image_data)
    print("shape of data is ",out.shape())
    print("output is : ", out)
    print("output shape is : ",out.shape())

def test_minpool_layer():
    """
        testing out minpooling layer
    """
    minpooling_layer = MinPool(kernal_size=5, stride=3, padding=2)
    image_data = Tensor(value=np.random.random(size=(10,40,40)))
    out = minpooling_layer.minpool(image_data)

    print("shape of data is ",out.shape())
    print("output is : ", out)
    print("output shape is : ",out.shape())
    


def test_avg_pool_layer():
    """
        testing out the average pooling layer
        similar as max and min pooling layer
    
    """
    avg_layer = AveragePool(kernal_size=3, stride= 2 , padding=1)
    image_data = Tensor(value= np.random.random(size= (10 , 32,32)))
    outcome = avg_layer.avgpool(x=image_data)
    print("data shape is : ", image_data.shape())
    print("data is : ", image_data )
    print("output data is : " , outcome)
    print("output data shape is :  ", outcome.shape())

def test_maxpool_color_channel():
    """
        image data will contain 3 color channel that is red , blue and green (RGB)
    """
    max_pool_layer3d = MaxPool3D(kernal_size=3, padding=1, stride=2)
    image_data = Tensor(value=np.random.random(size=(1,3,5,5)))
    print('image data is ',image_data)
    outcome = max_pool_layer3d.maxpool(x=image_data)
    print('ouput data is : ',outcome)
    print()
    print('output shape is ',outcome.shape())
    print('data shape is ',image_data.shape())

def testing_autograd_matrix_multiplication():
    obj1 = Tensor(value=[[1,2],[2,3]])
    obj2 = Tensor(value=[[3,6],[4,2]])
    obj3 = Tensor(value=[[1,3],[4,2]])
    obj4 = Tensor(value=[[2,4],[6,2]])

    out1 = obj1 * obj2
    out2 = out1 * obj3
    out3 = obj4 + out2

    final_out = out3.relu()
    final_out.backward()
    final_out.visualize_graph(filename='matric_ops')



def testing_autograd_for_linear_layer():
    """
        let us assume our linear layer has 100 nodes or perceptron.
    """
    x_data = Tensor(value= np.random.random(size=(10,3)) )
    layer1 = Layer(n_input=3 , n_out=10)
    layer2 = Layer(n_input=10,n_out=5)
    pred1 = layer1(x=x_data.data)
    out1 = pred1.tanh()
    out1.backward()
    pred2 = layer2(x=out1.data)

    y_pred = pred2.tanh()
    y_pred.backward()
    print('grad of out1 is ', out1.grad)
    print()
    print('grad of pred1 is ', pred1.grad)
    print()
    print('grad of pred2 is ', pred2.grad)
    print()
    print('grad of y_pred is ', y_pred.grad)
    y_pred.visualize_graph(filename="linear")


import torch
if __name__ == '__main__':
    # test_node()
    # test_linear_layers()
    # test_dense_layer()
    # test_conv_layer()
    # testing_autograd_basic_scalar_expr()
    # testing_autograd_basic_matrix_operation()
    # test_maxpool_layer()
    # test_minpool_layer()
    # test_avg_pool_layer()
    # test_maxpool_color_channel()
    #image_data = Tensor(value=np.random.random(size=(3,3,5,5)))
    # for batch , data in enumerate(image_data.data):
    #     print('data is ',data)
    #     print()
    #     print('data shape is ',data.shape)
    #     print()
    #     print()
    #obj1 = Tensor(value=np.random.random(size=(2,2)))
    # obj2 = Tensor(value=np.random.random(size=(2,3)))
    # c = obj1 * obj2 # (2,3)
    # print('c is ', c)
    # print('shape of c is ',c.shape())
    # print()
    # obj3 = Tensor(value=np.random.random(size=(2,3)))
    # out = c + obj3
    # print('out is ', out)
    # print('shape of out is ',out.shape())
    # print()
    # out.backward()
    # print('out grad is ', out.grad)
    # print('shape of c is ',out.grad.shape)
    # print()

    # print('obj3 grad is ',obj3.grad)
    # print('shape of obj3 grad is ',obj3.grad.shape)
    # print()

    # print(np.dot(c.data.T, out.grad))

    # out.visualize_graph()
    # testing_autograd_matrix_multiplication()
    # y_pred = Tensor(value=np.random.random(size=(1,10)) )
    # y_actual = Tensor(value=np.random.random(size=(1,10)) )
    # loss = y_actual.mse(y_pred)
    # print(loss)
    # print(loss.shape())
    # testing_autograd_for_linear_layer()

    node1 = Layer(n_input=10,n_out=3)
    print(node1)
    params = node1.parameters()
    # print(params)
    # print()
    data = np.random.random(size=(200,10))
    inputs = [list(map(Tensor, xrow)) for xrow in data]
    out = node1(x=inputs)
    # print(inputs)
    # print(len(inputs))
    print(len(out))


    # print()
    # layer = Layer(n_input=10,n_out=10)
    # print(layer)
    # params = layer.parameters()
    # print(params)
    # print(len(params))

   # print()
    # print()
    # dense = Dense(n_input=10 , list_layers=[10,20,10])
    # print(dense)
    # params = dense.parameters()
    # print(params)
    # print(len(params))




                  



    