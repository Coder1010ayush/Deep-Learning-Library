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
            params.grad = 0

    def parameters(self):
        return []
    



class NeuralNode(Module):

    def __init__(self,number_of_nodes,act = True) -> None:
        if number_of_nodes <= 0:
            raise ValueError(f"{number_of_nodes} can not be zero or negative!")
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
    
    def __repr__(self):
        return f"{'relu' if self.act else 'Linear'}Node({len(self.weigths)})"

        
class Layers(Module):

    def __init__(self, number_of_input_nodes,number_of_output_nodes, act=True) -> None:
        self.input_node = number_of_input_nodes
        self.outout_node = number_of_output_nodes
        self.act = act
        self.nodes = []

        for cnt in range(self.outout_node):
            self.nodes.append(NeuralNode(number_of_nodes=self.input_node,act=self.act))
        

    def parameters(self):
        return [p for n in self.nodes for p in n.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.nodes)}]"
    
    def __call__(self, x) :
        outcome = []
        for node in self.nodes:
            outcome.append(node(x))

        return outcome
    

class Dense(Module):

    def __init__(self,number_of_input, list_of_layers , act = True):
        self.act = act
        self.layers = []
        list_of_layers = [number_of_input] + list_of_layers
        for i in range(len(list_of_layers)-1):
            self.layers.append(Layers(number_of_input_nodes=list_of_layers[i],number_of_output_nodes=list_of_layers[i+1]))


    def __repr__(self):
        return f"Dense of [{', '.join(str(layer) for layer in self.layers)}]"
    

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __call__(self,x):
        outcome = []
        for layer in self.layers:
            outcome = layer(x)

        return outcome
