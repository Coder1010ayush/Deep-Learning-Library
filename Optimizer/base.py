"""
    this is a base class which wi;; acts as a blueprint 
    for all kind of gradient descent optimization algorithms.

"""
import sys
import math
import os
class BaseOptimizerClass:

    def __init__(self,learning_rate:float = 0.001,epcilon:float = 0.00001) -> None:  # simple gradient desecnt
        self.learning_rate = learning_rate
        self.epcilon = epcilon

    def __init__(self,learning_rate:float = 0.001, beta_1:float = 0.001, beta_2:float= 0.0001, epcilon:float= 0.00001): # adam
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epcilon = epcilon

    def __init__(self,learning_rate:float= 0.001, epcilon:float = 0.00001, weight_decay:float = 0.0001): # sgd
        self.learning_rate = learning_rate
        self.epcilon = epcilon
        self.weight_decay = weight_decay

    
    def __init__(self,learning_rate:float = 0.0001, epcilon:float = 0.00001, momentum:float = 0.001 , nesterov:bool= True):  # NAG prop
        self.learning_rate = learning_rate
        self.epcilon = epcilon
        self.momentum = momentum
        self.nesterov = nesterov


    def step(self, parameters):
        pass



class SGD(BaseException):

    def __init__(self, learning_rate, epcilon , weight_decay) -> None:
        super().__init__(learning_rate,epcilon,weight_decay)

    def step(self,parameters:list):
        accumulated_square_gradient  = 0
        for params in parameters:
            accumulated_square_gradient = self.decay_rate * accumulated_square_gradient + (1 - self.decay_rate) * params.grad^2
            regularization_term = self.weight_decay * params
            params = params - self.learning_rate* (params.grad + regularization_term) / (math.sqrt(accumulated_square_gradient) + self.epcilon)


class NAG(BaseException):
    
    def __init__(self,learning_rate:float = 0.0001, epcilon:float = 0.00001, momentum:float = 0.001 , nesterov:bool= True):
        super().__init__(learning_rate, epcilon, momentum , nesterov)

    def step(self,parameters:list):
        v = 0
        for params in parameters:
            v  = self.momentum * v -self.learning_rate * params.grad
            params = params - self.learning_rate * params.grad + v * self.momentum


class GD(BaseException):

    def __init__(self,learning_rate:float = 0.001,epcilon:float = 0.00001) -> None:
        super().__init__(learning_rate,epcilon)

    def step(self,parameters:list):
        for params in parameters:
            params = params - self.learning_rate*params.grad


class Adam(BaseException):

    def __init__(self,learning_rate:float = 0.001, beta_1:float = 0.001, beta_2:float= 0.0001, epcilon:float= 0.00001) :
        super().__init__(learning_rate,beta_1,beta_2,epcilon)

    def step(self,parameters:list):
        m1 = 0
        v = 0
        for params in parameters:
            m1 = m1*self.beta_1 - (1-self.beta_1)*params.grad
            m1 = m1/(1+self.beta_1**2)
            v = v*self.beta_2 - (1-self.beat_2) * params.grad
            v = v/(1+self.beta_2**2)
            params = params - (self.learning_rate*m1)/(math.sqrt(v)+self.epcilon)
