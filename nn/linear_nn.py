""""
    implementing linear layer using tensor class 
    that will calculate the gradient of each parameters 
    using the computational graph that will be drawn at runtime
    after defining the structure of the linear layer

"""
import os
import sys
from Tensor.tensor import Tensor
from Tensor.tensor import visualize_graph as vz
from Tensor.tensor import toNumpy
from Tensor.tensor import toTensor
import numpy as np
import pathlib
from activation.activation import relu, leaky_relu, tanh , sigmoid, gelu , selu
from initializer.xavier import Hei , LeCun , Xavier


# batched data into row and column data 
def flatten(data):
   
    if not isinstance(data, np.ndarray):
        raise ValueError("Input must be a NumPy ndarray")

    if data.ndim <= 2:
        return data.reshape(-1, data.shape[-1])

    flattened_data = np.array([])

    for row in data:
        if flattened_data.size == 0:
            flattened_data = flatten(row)
        else:
            flattened_data = np.concatenate((flattened_data, flatten(row)), axis=0)
    
    return flattened_data



"""
    implementation of linear layer will be very similar to pytorch 
"""

class Module: # this is the base class for the all neural network will be build beacuase it is the base class which have all the parameters assosiated to a model

    def zero_grad(self):
        for param in self.grad:
            param = 0.0

    def parameters(self):
        return []
    

"""
    implementing a single node in a linear layer 

"""
class Node(Module): 

    def __init__(self, n_in, activation_func:str = "") -> None:   # this class constructor only takes one parameter as n_in that is just for defining the weight and bias of the node 

        self.n_in = n_in
        self.act = activation_func
        # initialising the hei initializer for uniform initialization of weights 
        # it is generally recommended that we should not initialize all the weight of a node with random number or zeros or any constants.
        hei = Hei(n_in=self.n_in, uniform=True,shape=(self.n_in,1),shape_use=True)
        self.weight = Tensor(value=hei.initialize().data)
        # defining bias with zero
        self.bias = Tensor(value=np.random.random(size=(1,1))[0][0] ) 


    def parameters(self):
        yield self.weight
        yield self.bias

    def __call__(self , x):
        # calculate the outcome 
        outcome =  x * self.weight + self.bias
        return outcome

    # wrapper function => automatically called whenever the instance of the node will be tried out to be printed
    def __repr__(self) -> str:
        return f"Node{self.n_in}" 
    

    def info(self):
        info_dict = {}
        info_dict["n_in"] = self.n_in
        info_dict["weights"] = self.weight
        info_dict["bias"] = self.bias
        info_dict["activation_func"] = self.act
        return info_dict
    
"""
    implementing a single layer similar as defined in pytorch

"""
class Linear(Module):

    def __init__(self, in_feature:int = None , out_feature:int = None) -> None:
        self.in_feature = in_feature
        self.out_feature = out_feature
        # here intitialization of weights will be done on the basis of xavier rules
        hei = Xavier(n_in=self.in_feature , n_out=self.out_feature,uniform=True)
        self.weights = hei.initialize(shape=(self.in_feature, self.out_feature))
        # bias is choosen randomly anything using numpy random function 
        self.bias = Tensor(value=np.random.random(size=(1,1))[0][0] )


    def __repr__(self) -> str:
        return f"Linear(in_feature:{self.in_feature} , out_feature:{self.out_feature})"
    
    
    # this function is a generator 
    def parameters(self):
        yield self.weights 
        yield self.bias


    # forward pass method 
    def __call__(self , x):
        shape = list(x.data.shape)
        shape = shape[0:-1]
        shape.append(self.out_feature)
        if len(x.data.shape) != 2:
            x = Tensor(value=np.array(flatten(data=x.data)) )
        # print(x.data.shape)
        # print(self.weights.data.shape)
        # x = Tensor(value=x.data.T)
        out = x * self.weights  + self.bias
        final_out = Tensor(value=out.data.reshape(shape))
        return out,final_out
    
    def mse_loss(self, predictions, targets):
        # Mean Squared Error Loss
        diff = predictions - targets
        val =  (diff.square().sum())
        return val
    
    def backward(self, loss):
        # Backward pass to compute gradients
        loss.backward()
    
    def update_parameters(self, learning_rate):
        # Update weights and biases using gradients
        self.weights.data -= learning_rate * self.weights.grad
        self.bias.data -= learning_rate * self.bias.grad