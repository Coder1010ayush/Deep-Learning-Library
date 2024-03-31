"""
    this is a popular method to initialize the weights and biases of a 
    nueral networks. 
    It out performs than the random initialization of weights and biases.
    This approach of initialization is also known as Glorot initialization.

"""


from Tensor.matrix import Tensor
import math
import numpy as np
import os
import sys
import pydantic
import random

class Xavier:

    def __init__(self,n_in , n_out ,uniform:bool = False) -> None:
        self.n_in = n_in
        self.n_out = n_out
        self.uniform = uniform

    def initialize(self,shape):

        """Uniform initialization : 
                    this is how xaviar initializer works 
                    collection_set = U(-a , a)
                        where a is figured out by following formulae
                            a = ( 6/ (n_in + n_out) ) ** 0.2 
            Normal initialization : 

                    this is how normal xavier initializer is calculated
                    collection-set = U(0 ,sigma )
                            where a is figured out by following formulae
                            sigma = ( 2/ (n_in + n_out) ) ** 2

        """
        if self.uniform:  # this will call when the uniform xavier will be called as a  initializer 
            scale = np.sqrt( 6/(self.n_in + self.n_out) )
            return Tensor(value= np.random.uniform(size=shape, low=-scale , high= scale))

        else:  # this is spelled to call when the normal xavoer distribution is called as a initializer
            scale = np.square( 2/(self.n_in+self.n_out)  )
            return Tensor(value=np.random.normal(loc=0 , scale=scale , size=shape))



class Hei:

    def __init__(self, n_in, uniform,shape:tuple=None, shape_use : bool = False) -> None:
        self.n_in = n_in
        self.uniform = uniform
        self.shape_use = shape_use
        self.shape = shape

    def initialize(self):

        """
            Uniform distribution :
                collection_set = U(-a, a)
                    where a is figured out by following formulae
                        a = ( 2/ (n_in)) ** 0.5

            Normal distribution :
                collection_set  = U(0,a)
                    where a is defined by given relationship 
                    a  = (2/n_in) ** 0.5
        
        """
        scale = np.sqrt( 2/self.n_in) 
        if self.uniform:
            if self.shape_use:
                return Tensor(value=np.random.uniform(low=-scale , high=scale, size=self.shape))
            return Tensor(value=np.random.uniform(low=-scale , high=scale, size=(self.n_in, self.n_in)))

        else:
            if self.shape_use:
                return Tensor(value=np.random.normal(loc=0 , scale=scale, size=self.shape))
            return Tensor(value=np.random.normal(loc=0 , scale=scale, size=(self.n_in, self.n_in)))



class LeCun:  # nothing to much similar to hei initialization

    def __init__(self, n_in, uniform,shape:tuple=None, shape_use:bool=False):
        self.n_in = n_in
        self.uniform = uniform
        self.shape_use = shape_use
        self.shape = shape

    def initialize(self):

        """
            Uniform distribution :
                collection_set = U(-a, a)
                    where a is figured out by following formulae
                        a = ( 1/ (n_in)) ** 0.5

            Normal distribution :
                collection_set  = U(0,a)
                    where a is defined by given relationship 
                        a  = (1/n_in) ** 0.5
        
        """
        scale = np.sqrt( 1/self.n_in) 
        if self.uniform:
            if self.shape_use:
                return Tensor(value=np.random.uniform(low=-scale , high=scale, size=self.shape))
            return Tensor(value=np.random.uniform(low=-scale , high=scale, size=(self.n_in, self.n_in)))

        else:
            if self.shape_use:
                return Tensor(value=np.random.normal(loc=0 , scale=scale, size=self.shape))
            return Tensor(value=np.random.normal(loc=0 , scale=scale, size=(self.n_in, self.n_in)))



class ScaledStandaedDeviation: 

    def __init__(self, inputs:np.ndarray, uniform : bool = False , shape:tuple = None) -> None:
        self.shape = shape
        self.uniform = uniform
        self.inputs = inputs
        
    def feature_extractor(self):
        self.flatten_x = self.inputs.flatten()
        self.std = np.std(a=self.flatten_x)
        self.mean = np.mean(a=self.flatten_x)

    def initialize(self):
        """
            Uniform distribution :
                collection_set = U(-a, a)
                    where a is figured out by following formulae
                        a = ( 1/ (rv)) ** 0.5
                        rv = s**2/mean

            Normal distribution :
                collection_set  = U(0,a)
                    where a is defined by given relationship 
                        a  = (1/rv) ** 0.5
                        rv = s**2/mean
        
        """
        scale = np.sqrt( 1/self.rv ) 
        self.rv = self.std**2
        if self.uniform :
            return Tensor(value=np.random.uniform(low=-scale,high=scale,size=self.shape))
        

        else:
            return Tensor(value=np.random.normal(loc=0, scale=scale, size=self.shape))
    

