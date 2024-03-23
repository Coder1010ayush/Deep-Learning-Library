"""
    this class implements all the kinds of Linear Neural Networks.
    such as : Single layer , Multi layer 

"""
import os
import sys
import random
from Tensor.matrix import Tensor
from Optimizer.base import BaseOptimizerClass,SGD,Adam,NAG

__all_activation__functions__ = ["relu","sigmoid","leaky_relu","selu","elu","glu","tanh","swish"]

class Module:

    def zero_grad(self):
        
        for params in self.parameters():
            self.grad = 0

    def parameters(self):
        return []
    



class NeuralNode(Module):

    def __init__(self,number_of_nodes,act = True) -> None:
        self.act = act
        self.number_of_nodes = number_of_nodes
        self.weigths = []
        for i in range(self.number_of_nodes):
            self.weigths.append(Tensor(random.normalvariate(mu=0,sigma=1)))
        self.bias = Tensor(0)


    def __call__(self,x):
        accumulated_sum = 0
        for weight , bias in zip((self.weigths,x),self.bias):
            accumulated_sum += weight.data * bias.data
        if self.act:
            loss = accumulated_sum.relu()
        else:
            loss = accumulated_sum
        return Tensor(loss)
    

    def parameters(self):
        return self.weigths + [self.bias]
    
    def __repr__(self) -> str:
        # f"( \nNueralNode \nParameters {(self.weigths)},{([self.bias])} \n)"
        return f"Node({self.number_of_nodes})"
        
class Layers(Module):

    def __init__(self, number_of_input_nodes,number_of_output_nodes, act=True) -> None:
        self.input_node = number_of_input_nodes
        self.outout_node = number_of_output_nodes
        self.act = act
        self.nodes = []

        for cnt in range(self.outout_node):
            self.nodes.append(NeuralNode(number_of_nodes=self.input_node,act=self.act))
        

    def parameters(self):
        self.params = []
        for node in self.nodes:
            self.params.extend(node.parameters())
        return self.params
    
    def __repr__(self) -> str:
        return f"Layer{self.input_node,self.outout_node}\n"
    
    def __call__(self, x) :
        outcome = []
        for node in self.nodes:
            outcome.append(node(x))

        return outcome
    

class Dense(Module):

    def __init__(self,number_of_input, list_of_layers , act = True):
        self.input_nodes = number_of_input
        self.list_of_layers = list_of_layers
        self.act = act

        self.layers = []
        for item in self.list_of_layers:
            self.layers.append(Layers(number_of_input_nodes=self.input_nodes,number_of_output_nodes=item)) # defining all the layer and appending to the self.nodes list 
        


    def __repr__(self) -> str:
        return f"Dense(\n{self.layers}\n)"
    

    def parameters(self):
        self.params = []
        for layer in self.layers:
            self.params.extend(layer.parameters())
        return self.params
    
    def __call__(self,x):
        outcome = []
        for layer in self.layers:
            outcome = layer(x)

        return outcome
