"""

    implementing all the single neural node , single layer of nodes and a danse layer
    this is based on numpy for fast calculation and wrapper class Tensor.
    this works for n dimensional data.
"""


import numpy as np
import os
import sys
from Tensor.matrix import Tensor
from initializer.xavier import Hei,Xavier

"""
    similar to pytorch 
    defining Module class which have two function 
        01. zero_grad -- for initializing all the grad of weight and bias with zero
        02. parameters -- this is just for initializing parameters with an empty list
    this class acts as a base class for all other class Node , Layer and Dense.
"""
class Module:

    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0

    def parameters(self):
        return []
    
"""

this class implements working of a single node with a given value that is the number of weights 
and there is a single bias assosiated to this node.
this will use in a single Layer formation

"""
class Node(Module):

    def __init__(self,n_input) -> None:
        self.n_input = n_input
        hei = Hei(n_in=n_input,uniform=True,shape=(1,1))
        self.w = None
        weigths = []
        self.bias = 0
        for i in range(self.n_input):
            x = hei.initialize().data[0][0]
            weigths.append(x)
        self.w = np.array(object=weigths)

    def __repr__(self) -> str:
        return f"Node({self.n_input})"
    
    def __call__(self,x):
        out = np.dot(x , self.w) + self.bias
        return out
    
    def parameters(self):
        return self.w.flatten().tolist() + [self.bias]
    

"""

    this class implements a layer of given n_input nodes.
    each node has n_outs number of bias 
    each node has n_input times n_outs of weights 
    this is used for Dense Network formation

"""
class Layer(Module):

    def __init__(self,n_input , n_out) -> None:
        self.n_input = n_input
        self.n_outs = n_out
        xv = Xavier(n_in=self.n_input, n_out=self.n_outs,uniform=True)
        self.bias = xv.initialize(shape=(self.n_outs)).data
        self.w = xv.initialize(shape=(self.n_input,self.n_outs)).data

    def __repr__(self) -> str:
        return f"Layer of [{', '.join('LinearNode'+str(self.n_input) for n in range(self.n_input ))}]"
    
    def parameters(self):
        params = []
        w = self.w.flatten().tolist()
        b = self.bias.tolist()
        return w + b
    
    def __call__(self,x):
        out = np.dot(x, self.w)
        return out
    
"""
    this class implements a list of layers that is given as list_of_layer
    the parameter list_of_layer contains the output produced by each layer 
    for example first element represents the number of column generated by first layer. similarly for each elements.
    A dense networks has no its own weigths and bias but each layer contained has its own weights and biases
    this class is a general representation of multi layer perceptron or a dense layer defined in Tensorflow!

"""
class Dense(Module):

    def __init__(self, n_input:int , list_of_layer:list) -> None:
        self.n_input = n_input
        self.w = None
        self.b = None
        layer_segment = [self.n_input] + list_of_layer
        self.layers = [Layer(layer_segment[i], layer_segment[i+1]) for i in range(len(list_of_layer))]


    def __repr__(self) -> str:
        return f"Dense [{', '.join(str(layer) for layer in self.layers)}]"
    
    def __call__(self , x):
        # exploring each layer output one previous layer is input for current layer
        for layer in self.layers:
            
            x = layer(x=x)
           # print('shape of x is ', x.shape)

        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params