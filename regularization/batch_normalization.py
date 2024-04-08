"""
    this class implements the batch normaliaztion
    for input data.

"""
import os
import sys
import pathlib
import numpy as np
from Tensor.tensor import Tensor,visualize_graph,toNumpy,toTensor


class BatchNormalization:

    def __init__(self) -> None:
        self.batch_size = None
        self.current_mean = None
        self.current_var = None
        self.gamma = None
        self.beta = None
        self.epsilon = None
        self.x_normalised = None


    def initializer(self , input_shape):
        self.gamma = np.zeros_like(input_shape)
        self.beta = np.zeros_like(input_shape)
        self.current_mean = np.zeros_like(input_shape)
        self.current_var = np.zeros_like(input_shape)


    def __call__(self , x, mode="train" , learning_rate = 0.001): # mode should be by default training rather than testing.
        # including both forward and backward pass also
        out = None
        if mode == 'train':
            # Compute batch statistics
            self.batch_size = x.data.shape[0]
            batch_mean = np.mean(x.data, axis=0)
            batch_var = np.var(x.data, axis=0)
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            # Normalize input
            self.x_normalized = (x.data - batch_mean) / np.sqrt(batch_var + self.epsilon)
            
            # Scale and shift normalized input
            out = Tensor(value=self.gamma * self.x_normalized + self.beta,operation="Backward<BN>", subset=(x,) )
            
        elif mode == 'test':
            self.x_normalized = (x.data - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            
            # Scale and shift normalized input
            out = Tensor(value=self.gamma * self.x_normalized + self.beta )
        
        else:
            raise NameError("wrong mode is given only train and test is supported!")
        

        def _backward():
            dgamma = np.sum(out.data * self.x_normalized, axis=0)
            dbeta = np.sum(out.data, axis=0)
            
            # Compute gradient of input
            dx_normalized = out.data * self.gamma
            dvar = np.sum(dx_normalized * (self.x_normalized * -0.5) / np.sqrt(self.running_var + self.epsilon), axis=0)
            dmean = np.sum(dx_normalized * -1 / np.sqrt(self.running_var + self.epsilon), axis=0) + dvar * np.mean(-2 * (self.x_normalized - self.running_mean), axis=0)
            dx = dx_normalized / np.sqrt(self.running_var + self.epsilon) + dvar * 2 * (self.x_normalized - self.running_mean) / self.batch_size + dmean / self.batch_size
            out =out - Tensor(value=learning_rate*dx )
            self.gamma -= learning_rate*dgamma
            self.beta -= learning_rate *dbeta

        out._backward = _backward
        
        return out
