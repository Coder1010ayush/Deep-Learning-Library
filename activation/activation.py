""""
    this class implements all kinds of activation function that will be utilising in building neural network architecture.
    all the loss function will be included here.

"""
import math 
import numpy as np
from typing import overload
from Tensor.tensor import Tensor
from Tensor.tensor import toNumpy
from Tensor.tensor import toTensor
from Tensor.tensor import visualize_graph

"""
    let us proceed with inheriting the Tensor class so that we can overload all 
    the activation functons and loss function both.

"""

def relu(self):
        """
            matrhematical formulation :
                f(x)>= 0 : f(x)
                f(x) <0 : 0

            derivation is as below:
                if f'(x) = 1 , if f(x) >= 0
                else f'(x) = 0 , if f(x) < 0
        
        """
        # there are two cases for covering 
        # one is when self is just a single value 
        outcome = None
        def filter(x):  # making a filter so that if element is greater than 1 than its 1 otherwise 0
            return np.maximum(0,x)
        if isinstance(self.data , (int , float)):
            outcome = Tensor(value=max(0, self.data), operation="Backward<relu>",subset=(self,) )
            def _backward():
                loc_res = 1
                if self.data > 0:
                    loc_res = 1
                else:
                    loc_res = 0
                self.grad += loc_res * outcome.grad
            outcome._backward = _backward

        # second is when self is a matrix or numpy ndarray
        elif isinstance(self.data , np.ndarray):
            # print(self.data) # both are same if all elements are greater than zero otherwise it will be different!
            # print(filter(x=self.data))
            outcome = Tensor(value=filter(x=self.data), operation="Backward<relu>", subset=(self,))
            loc_res = np.where(self.data > 1 ,1 ,0)
            def _backward():
                self.grad += loc_res * outcome.grad
            outcome._backward = _backward
        else:
             raise ValueError("unsupported data type is encountered!")

        return outcome

def tanh(self):
    """
        mathematical formulation :
            f(x) = (e{x} - e{-x}) / (e{x} + e{-x})
        derivative of this function is :
            f'(x) = 1 - f(x)**2
     
    """
    outcome = None
    # backpropogation occurs
    def _backward():
        self.grad += (1- outcome.data**2) * outcome.grad

    # handling single scalar input 
    if isinstance(self.data , (int , float)):
        outcome = Tensor(value=(   (np.exp(2*self.data) - 1)/(np.exp(2*self.data ) -1) ), subset=(self,),operation="Backward<tanh>")
        outcome._backward = _backward

    # handling tensor object input 
    elif isinstance(self.data , np.ndarray):
        outcome = Tensor(value=(   (np.exp(2*self.data) - 1)/(np.exp(2*self.data ) -1)  ), subset=(self,),operation="Backward<tanh>")
        # print(outcome)
        outcome._backward = _backward 

    # nothing else is supported!
    else:
        raise ValueError("unsupported data type is encountered!")
    return outcome

def sigmoid(self):
        """
            mathematical formulation :
                f(x) = 1/(1+e{-x})
            derivative of this function is :
                f'(x) = sigmoid(x) * (1-sigmoid(x))        
        """
        outcome = None
        # defining backpropogation for the sigmoid function
        def _backward():
             self.grad += (outcome.data * (1-outcome.data))*outcome.grad  # chain rule is applied
        if isinstance(self.data , (int , float)):
             outcome = Tensor(value= 1/(1+np.exp(self.data)),subset=(self ,), operation="Backward<Sigmoid>")
        
        elif isinstance(self.data , np.ndarray):
             outcome = Tensor(value=1/(1+ np.exp(-self.data)),subset=(self,),operation="Backward<Sigmoid>")

        
        outcome._backward = _backward
        return outcome


def leaky_relu(self,lamda=0.001): 
        """
        function formula: 
            f(x)  = {
                x : x >= 0
                lamda*x : x < 0 
            }

        derivative for this function is :
            f'(x) = {
                1 : x>=0
                0 : x<0
            }
        """
        outcome = None
        def filter(x):
            return np.maximum(0,x)
        
        def _backward():
            mask = filter(self.data)
            self.grad += (lamda*mask) * outcome.grad

        if isinstance(self.data , (int , float)):
            def _backward():
                if self.data >= 0:
                     self.grad += 1*outcome.grad
                else:
                     self.grad += lamda * outcome.grad
            if self.data >= 0:
                outcome = Tensor(value=self.data,subset=(self,), operation="Backward<Leaky_Relu>")
            else:
                outcome = Tensor(value=lamda*self.data,subset=(self,), operation="Backward<Leaky_Relu>")
            outcome._backward = _backward

        elif isinstance(self.data , np.ndarray):
            outcome = Tensor(value=np.where(self.data > 0 , self.data , lamda*self.data) , operation="Backward<Leaky_Relu>",subset=(self,))
            outcome._backward = _backward

        else:
             raise ValueError("not supported data type is encountered!")
        return outcome


def swish(self): 
        """ 
            function formulae : 
                f(x) : x * sigmoid(x)
            differentiation of f(x) with respect to x = 
                f'(x) = x (sigmoid(x)-(1-sigmoid(x) ) + 1 * sigmoid (x)
        """ 
        def func(x):
             return 1/(1+np.exp(x))            
        outcome = None
        if isinstance(self.data , (int , float)):
             outcome = Tensor(value= self.data * func(self.data),subset=(self,), operation="Backward<Swish>")
        
        elif isinstance(self.data , np.ndarray):
             outcome = Tensor(value= self.data * func(x=self.data), subset=(self,), operation="Backward<Swish>")


        else:
             raise ValueError("not supported data type is encountered!")
        
        def _backward():  # defining backpass for swish function 
             self.grad += self.data * (outcome.data * (1 -outcome.data ) ) + 1 + outcome.data
        outcome._backward = _backward

        return outcome


def gelu(self):
    """
        GELU(x)=0.5*x* (1+Tanh(2/π*  (x+0.044715*x3)  )  )  
        here 2/pi is in square root
    """

    outcome = None
    res = None
    # (np.exp(2*self.data) - 1)/(np.exp(2*self.data ) -1)
    beta = 2/3.14 * (self.data + 0.044715 * self.data * self.data * self.data)
    if isinstance(self.data , (int , float)):
         res = (np.exp(2*beta) - 1)/(np.exp(2*beta ) -1)
         outcome = Tensor(value=0.5*self.data * res, subset=(self,),operation="Backward<Gelu>")

    elif isinstance(self.data , np.ndarray):
         res = (np.exp(2*beta) - 1)/(np.exp(2*beta ) -1)
         outcome = Tensor(value= 0.5 * self.data * res, subset=(self,),operation="Backward<Gelu>")
    
    else:
         raise ValueError("unsupported data type is encountered!")
    
    def _backward():
         self.grad += (0.5*res + 0.5*self.data * (1-beta*beta)) * outcome.grad

    outcome._backward = _backward
    return outcome

def selu(self):
    scale = 1.0507009
    alpha = 1.6732
    """
        mathematical function :
            f(x) = scale * (max(0, x) + min(0 , alpha * (exp(x) - 1)))

        derivative of this function is :
            f'(x) = {

                    derivative =    scale (1 + alpha * exp(x) )     if x >= 0 
                    derivative =    scale (0 + 0 ) => 0                if x < 0

                    }
    """
    outcome = None
    scale = 1.0507009
    alpha = 1.6732

    def calculate_first_half(x):
         return np.maximum(0 , x)
    
    def calculate_second_half(x):
         return np.minimum(0 , alpha * (np.exp(x) - 1))


    if isinstance(self.data , (int , float)):
         outcome = scale * (max(0, self.data) + min(0 , alpha*(math.exp(self.data - 1))))
         outcome = Tensor(value= outcome, operation="Backward<Selu>", subset=(self,))
         def _backward():
              
            if self.data >= 0:
                self.grad += scale * (1+ alpha* math.exp(self.data))

            else:
                self.grad += 0
         outcome._backward = _backward

    elif isinstance(self.data , np.ndarray):
        outcome = Tensor(value = scale*(np.max(0,self.data)+ np.min(0 , np.exp(self.data)-1)) , subset=(self, ) , operation="Backward<Selu>")

        def _backward():
            # backpropogation for selu function
            self.grad += scale * (1 + alpha * (np.exp(self.data )))

        outcome._backward = _backward

    
    else:
         raise ValueError("unsupported data types is encountered!")
    
    return outcome



def elu(self,alpha=0.001):  # implementing backpropogation for elu activation function 
        """
        function formulae :
                R(z) = {
                    z ,z>=0
                    alpha.(e^z - 1) , z<0
                }
        differentiation of f(x) with respect to x = 
                R'(z) = {
                    1  , z>=0
                    alpha.e^z   z<0
                    
                }
        """
        def filter(x):
            return np.where(x>0,x,alpha*(np.exp(x)-1))
        
        outcome = None
        def _backward():
             self.grad  += alpha * np.exp(self.data) * outcome.grad

        if isinstance(self.data , (int , float)):
             outcome = Tensor(value= filter(self.data), subset=(self,), operation="Backward<Elu>")
             
        elif isinstance(self.data , np.ndarray):
             outcome = Tensor(value=filter(x=self.data), subset=(self , ), operation="Backward<Elu>")

        
        else:
             raise ValueError("unsupported data type is encountered!")
        outcome._backward = _backward
        return outcome

"""
    Below it all the loss funcition would be implemented 
        01. MSE Loss function
        02. Softmax Loss function
        03. Binary Cross Entropy Loss function
        04. Categorical Cross Entropy Loss function
        05. Hubour's Loss function
        06. RMSE

"""

class MeanSquareError:
     
    """
        this class implements the forward and backpropogation for the mean square loss function 
        this is mainly used for regression problems.
    """

    def __call__(self,prediction, other):
        """
            calculating loss using mse formulae :
            Mathematical formulae =>
                f(x) =summation ( (other - self)**2 )

            Internally it will handle all the backpropogation things using the computational graph.
        """
        diff = prediction - other
        val = diff.square().sum()
        return val
       

class Softmax:
     
     """
        this class implements the forward and backward propogation of the softmax function.
        this loss function is generally used for the squasshing the elements of a matrix between zero and one like as propability.
     """

     def __call__(self, prediction):
          """
            calculating the probabilities for given prediction
            mathematical formulation =>
                f(x) = e{xi} / sum(e{xj})  
          """
          den = prediction.exp().sum()
          val = prediction/ den
          return val
     
class BinaryCrossEntropy:
     
     """
        this class implements the forward and backpropogation for the binary cross entropy for classification propblems.
     """

     def __call__(self, predictions, targets) :
          """
            calcultating the loss for prediction 
          """
          epsilon = Tensor(value=1e-15 ) # to avoid taking the log of 0
          out = ((targets * (predictions + epsilon).log()  ) + (Tensor(value=1) - targets) * ((Tensor(value=1) - targets) + epsilon).log() ).mean()
          return out
     


class CategoraicalCrossEntropy:
     
     """
        this class implements the forward and backward pass for the categorical cross entropy for the multi class classification problems.
     """
     def __call__(self, prediction , targets):
          epsilon = Tensor(value=1e-15 ) # to avoid taking the log of 0
          out = -(targets * (prediction + epsilon).log().mean(axis = 1) )
          return out
