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
import math

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

    def __init__(self, in_feature:int = None , out_feature:int = None,learning_rate=0.01,decay_rate=0.1) -> None:
        self.in_feature = in_feature
        self.out_feature = out_feature
        # here intitialization of weights will be done on the basis of xavier rules
        hei = Xavier(n_in=self.in_feature , n_out=self.out_feature,uniform=False)
        self.weights = hei.initialize(shape=(self.in_feature, self.out_feature))
        # bias is choosen randomly anything using numpy random function 
        self.bias = Tensor(value=0.0)
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.best_loss = np.inf
        self.patience = 3  # Number of epochs to wait before early stopping
        self.wait = 0



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
        
        out = ( x * self.weights  ) + self.bias 
        return out
    
    def mse_loss(self, predictions, targets):
        # Mean Squared Error Loss
        shape = list(targets.data.shape)
        shape = shape[0:-1]
        shape.append(self.out_feature)
        if len(targets.data.shape) != 2:
            targets = Tensor(value=np.array(flatten(data=targets.data)) )
        diff = predictions - targets
        val =  (diff.square().sum())
        return val
    
    def backward(self, loss):
        # Backward pass to compute gradients
        loss.backward()
    
    def update_parameters(self, learning_rate,epoch):
        self.weights.grad = np.clip(self.weights.grad, -1, 1)
        self.bias.grad = np.clip(self.bias.grad, -1, 1)
        # Update weights and biases using gradients
        self.weights.data -= learning_rate * self.weights.grad
        #print('bias is ', self.bias)
        self.bias.data -= learning_rate * self.bias.grad
        self.learning_rate *= (1. / (1. + self.decay_rate * epoch))  # Learning rate decay


    # def check_early_stopping(self, validation_loss):
    #     if validation_loss < self.best_loss:
    #         self.best_loss = validation_loss
    #         self.wait = 0
    #     else:
    #         self.wait += 1
    #         if self.wait >= self.patience:
    #             return True  # Stop training
    #     return False  # Continue training


    
    def info(self):
        info_dict = {
            "in_feature": self.in_feature,
            "out_feature": self.out_feature,
            "weights": self.weights,
            "bias": self.bias,
            "learning_rate": self.learning_rate,
            "decay_rate": self.decay_rate
        }
        return info_dict

    def zero_grad(self):
        self.weights.grad = 0.0
        self.bias.grad = 0.0
    

"""
    implementing a dense network of containing a list of single linear layer 

"""
class Sequential:

    def __init__(self, list_of_layers:list = []) -> None:
        """
            list_of_layers would be in the form of a list of the tuple in which each tuple
            contain two elements in which first element is the in_feature and second element of 
            the tuple is the out_feature of that layer.

        """
        self.number_of_layers = len(list_of_layers)
        self.list_of_layers = list_of_layers
        self.layers = []
        # creating a list of layers according to the given list
        for i in list_of_layers:
            in_feature = i[0]
            out_feature = i[1]
            layer = Linear(in_feature=in_feature , out_feature=out_feature)
            self.layers.append(layer)


    # way of showing how the model looks like 
    def __repr__(self) -> str:
        string = []
        for i in self.list_of_layers:
            str1 = "Linear"+str(i)
            string.append(str1)

        pretty_str = "("

        for i in string:
            pretty_str += "\n"+"   "+i

        pretty_str += "\n" + ")"
        return pretty_str
    

    def __call__(self , x):
        out = None
        for layer in self.layers:
            x = layer(x)
        return x

    def mse_loss(self, predictions, targets):
        # Mean Squared Error Loss
        diff = predictions - targets
        val =  (diff.square().sum())
        return val
    
    def backward(self, loss):
        # Backward pass to compute gradients
        loss.backward()

    
    def parameters(self):
        for layer in self.layers:
            yield from layer.parameters()

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()