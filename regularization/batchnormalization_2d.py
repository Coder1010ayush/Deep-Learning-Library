""""
    this class impelemnts the batch normalization for the 2d or 3d data only.
    input data must have at max 3 dimension.
"""
import os
import numpy as np
from initializer.xavier import Xavier, Hei, LeCun
from Tensor.tensor import Tensor
from Tensor.tensor import visualize_graph as vz
from Tensor.tensor import toNumpy , toTensor
from activation.activation import tanh, sigmoid , leaky_relu , relu , gelu ,selu


class BatchNorm1d:

    def __init__(self, episilon = 1e-15, momentum = 0.001, leaning_rate = 0.001) -> None:
        
        self.epsilon =Tensor(value= episilon )
        self.momentum = Tensor(value= momentum )
        self.current_mean = None
        self.current_var = None
        self.gamma = None
        self.beta = None
        self.normalized_x = None
        self.learning_rate = leaning_rate

    def initialize_parameters(self , number_of_features):
        # initializing all the parameters 
        # initialized using Hei methode

        self.number_of_feature = number_of_features
        xv = Hei(n_in=self.number_of_feature, uniform=True, shape=(self.number_of_feature),shape_use=True)
        self.current_mean = xv.initialize()
        self.current_var = xv.initialize()
        self.gamma = Tensor(value= np.ones(shape=(self.number_of_feature)))
        self.beta = Tensor(value= np.zeros(shape= (self.number_of_feature)))

    # every class have this method
    def parameters(self):
        yield self.gamma
        yield self.beta


    """
        forward propogoation is written using all the Tensor object so no need to write backward pass explicitly 
        it will be handled automatically in Tensor.tensor class for each tensor object simultaneously.
    """
    def forward(self , x):
        self.batch_size = x.data.shape[0]
        outcome = None
        # calculating batch_mean and batch variance 
        batch_mean = Tensor(value= np.mean(x.data , axis=0) )
        batch_var = Tensor(value= np.var(x.data , axis= 0) ) 

        # changing the current mean and current var using these parameters
        self.current_mean = self.current_mean * self.momentum + (Tensor(value=1.0) - self.momentum) * batch_mean
        self.current_var = self.current_var * self.momentum + (Tensor(value=1) - self.momentum) * batch_var
        
        # normalized data points
        self.normalized_x = (x - batch_mean) / ( (batch_var + self.epsilon).sqrt() )

        outcome = self.normalized_x * self.gamma + self.beta

        return outcome

    """
        updating the parameter according to learning rate given and using decay rate also.
    """

    def updateParameters(self ,epoch , decay_rate = 0.001):
        # clipping the gradient vlaue between -1 to 1 to avoid gradient exploding
        self.gamma.grad = np.clip(self.gamma.grad, -1, 1)
        self.beta.grad = np.clip(self.beta.grad, -1, 1)

        # updating the gamma and beta paramter using learning rate 
        self.gamma.data -= self.learning_rate * self.gamma.grad
        self.beta.data -= self.learning_rate * self.beta.grad
        # Learning rate decay
        self.learning_rate *= (1. / (1. + decay_rate * epoch)) 

