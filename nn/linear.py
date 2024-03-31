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
import random


class Module:   # a base class for all nueral network architecture because every nn must have zero_grad and parameters function

    def zero_grad(self):
        
        for param in self.parameters():
            param.grad = 0.0

    
    def parameters(self):
        return []
    
class Node(Module):   # implementing a single node with some weights and bias given 

    def __init__(self, n_input):
        # initialising the weight of the node using Hei method
        # more references in xavier.py
        hei = Hei(n_in=n_input,uniform=True,shape=(1,1))
        self.weights = [Tensor(value=hei.initialize().data[0][0]) for _ in range(n_input)]
        self.bias = Tensor(0)



    def parameters(self):
        # print('weigth is : ', self.weights)
        # print('bias is : ',self.bias)
        return self.weights + [self.bias]
    

    def __repr__(self):
        return f"LinearNode{len(self.weights)}"
    

    
    
    def __call__(self,x):
        outcome = []
        weight_list = []
        for item in self.weights:
            weight_list.append(item.data)
        weight_matrix = Tensor(value=np.array(object=weight_list))
        for row in x:
            # converting row tensor list into tensor matrix
            # print('start! from here !')
            # print(row)
            data_list = []
            for item in row:
                data_list.append(item.data)
            data_matrix = Tensor(value=np.array(object=weight_list))
            
            val = Tensor(value=np.dot(data_matrix.data.T , weight_matrix.data))
            outcome.append(val)
        if len(outcome) == 1:
            return outcome[0]
        return outcome
            



class Layer(Module):   # implementing a single layer of nodes 

    def __init__(self, n_input, n_out):
        self.n_in = n_input
        self.n_outs = n_out
        self.nodes = []

        for i in range(0, self.n_outs):
            self.nodes.append(Node(n_input=n_input))
    
    
    def parameters(self):
        self.params = []
        for node in self.nodes:
            self.params.extend(node.parameters())
        return self.params
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.nodes)}]"
    
    """
        for optimising this step we have to deep dive into some other special optimised linear algebra library and some hardware side maily parallel computinng concept. 
        this works for 100*100 matrix fine. may be optimised it using some kind of sparse matrix concept also . for now it is sufficient.

    """
    def __call__(self, x):
        outcome = []
        for node in self.nodes:
            val = node(x)
            outcome.extend(val)
        out = np.array(object=outcome)
        if self.n_outs == 1:
            out = out.reshape(len(x),)
        else:
            out = out.reshape(len(x),self.n_outs) 
         # output column must be same as n_outs and number of columns in input must be same as n_input than reshaping will happen correctly otherwise error will be raised!
        if len(outcome) == 1:
            # print('single element')
            return outcome[0]
        else:
            # print('length of outcome list is ', len(outcome))
            # print('multilple elements')
            # print()
            return out.tolist() 
       
        

        
class Dense(Module):

    def __init__(self, n_input, list_layers:list):

        total_size = [n_input] + list_layers
        self.layers = [Layer(total_size[i], total_size[i+1]) for i in range(len(list_layers))]
        
    def parameters(self):
        self.params = []
        for layer in self.layers:
            self.params.extend(layer.parameters())
        return self.params        # returning a list of weights and bias 
    

    def __repr__(self):
        return f"Dense of [{', '.join(str(layer) for layer in self.layers)}]"
    

    def __call__(self, x):
        # calling each layer 
        outcome = []
        cnt = 0
        for layer in self.layers:
            
            x = layer(x=x)
            
        return x


    


