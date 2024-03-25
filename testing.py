import os
import sys
import numpy as np
import json
from Tensor.matrix import Tensor
from Tensor.utility import visualize_graph
from nn.cnn import conv
from nn.linear import NeuralNode , Layers , Dense
from nn.pooling import MaxPool, MinPool , AveragePool


def test_linear_layers():
    """
        testing linear layers from nn.linear
    """
    layer = Layers(number_of_input_nodes=10 , number_of_output_nodes= 20)
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
    node = NeuralNode(number_of_nodes=10)
    print('node is : ',node)
    print('parameters are : ', node.parameters())

    node = NeuralNode(number_of_nodes=6)
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
    pass


def test_minpool_layer():
    pass


def test_avg_pool_layer():
    pass




if __name__ == '__main__':
    test_node()
    test_linear_layers()
    test_dense_layer()
    test_conv_layer()
