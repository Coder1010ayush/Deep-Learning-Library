""""
    implementation of Single node with some weights and bias.
    implementation of Single Layer with any number of Single Nodes 
    implementation of a whole connected dense layer according to given information about layers.


    Structure will be similar as pytorch,
        there will a Module class that will be inherited by each single class , Module class defined the weights and biases.
        it incorporates a parameters function similar to pytorch that will returns all the weights and biases of the network. 

"""
from Tensor.matrix import Tensor
from initializer.xavier import Xavier , ScaledStandaedDeviation , LeCun , Hei
from Optimizer.base import BaseOptimizerClass,NAG, SGD, Adam
import os
import collections
import sys
import numpy as np


class Module:   # a base class for all nueral network architecture because every nn must have zero_grad and parameters function

    def _zero_grad(self):
        
        for param in self.parameters():
            param.grad = 0.0

    
    def parameters(self):
        return []
    
class Node(Module):   # implementing a single node with some weights and bias given 

    def __init__(self, n_input ):
        self.nin = n_input
        self.w = []
        xev = Xavier(n_in=1, n_out=1)
        for i in range(self.nin):
            # x = np.random.random(size=1)
            x = xev.initialize().data[0]
            self.w.append(x[0])
        self.weights = Tensor(value=self.w)
        self.bias = Tensor(value=0)


    def parameters(self):
        return [self.weights] + [self.bias]
    

    def __repr__(self):
        return f"LinearNode{(self.weights.data.shape)}"
    
    def __call__(self,x):
        activation_output = 0
        for weight_i , x_i in zip(self.weights.data, x):
            # print()
            # print(weight_i, '      ',x_i)
            out = weight_i * x_i
            activation_output = activation_output + (self.bias.data + out)
        # print(activation_output)
        # print(type(activation_output))

        return Tensor(value=activation_output)  # return a single tensor with multiplication of weights and input data with addition of biases
    

   


class Layer(Node):   # implementing a single layer of nodes 

    def __init__(self, n_input, n_out):
        self.n_in = n_input
        self.n_outs = n_out
        self.nodes = []

        for i in range(0, self.n_outs):
            self.nodes.append(Node(n_input=n_input))
    
    def parameters(self):
        self.params = []
        self.weigths = []
        self.bias = []
        for node in self.nodes:
            weigth , bias = node.parameters()
            self.weigths.append(weigth.data)
            self.bias.append(bias.data)
        weight_tensor = Tensor(value=np.array(object=self.weigths).reshape(self.n_outs,self.n_in))
        bias_tensor = Tensor(value=np.array(object=self.bias).reshape(self.n_outs))

        return [weight_tensor] + [bias_tensor]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.nodes)}]"
    
    """
        for optimising this step we have to deep dive into some other special optimised linear algebra library and some hardware side maily parallel computinng concept. 
        this works for 100*100 matrix fine. may be optimised it using some kind of sparse matrix concept also . for now it is sufficient.

    """
    def __call__(self, x):
        shape_of_x = x.shape
        #print(shape_of_x)
        x = x.flatten()
        # print(x.shape)
        col = shape_of_x[-1]
        row = int(x.shape[0]/col)
        x = x.reshape(row, col)
        outcome = []
        for node in self.nodes:
            for i in range(0, row):
                data = x[i]
                out = node(data).data
                outcome.append(out)
        
        reshaped_list = list(shape_of_x)
        reshaped_list[-1]  = self.n_outs
        #print(reshaped_list)
        return Tensor(value=np.array(object=outcome).reshape(reshaped_list))
        

        
class Dense(Layer):

    def __init__(self, n_input, list_layers:list):
        self.n_input = n_input
        self.n_layers = len(list_layers)
        self.layers = []
        for i in range(0, self.n_layers):
            self.layers.append(Layer(n_input=n_input,n_out=list_layers[i]))
        
    def parameters(self):
        self.params = []
        for layer in self.layers:
            self.params.append(layer.parameters())
        return self.params          # returning a list of a list that contains a tensor of weights and a tensor of bias  ---> internal list has 2 length because it has a weight and a bias.


    def __repr__(self):
        return f"Dense of [{', '.join(str(layer) for layer in self.layers)}]"
    

    def __call__(self, x):
        # calling each layer 
        # outcome = []
        cnt = 0
        for layer in self.layers:
            if cnt==0:
                out = layer(x).data
            else:
                out = layer(out).data
        return Tensor(value=out)


    


