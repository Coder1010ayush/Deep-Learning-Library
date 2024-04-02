"""
    implementing a general Tensor class which is built on the numpy array from skretch.
    this is very similar to pytorch class (inspiration) supported by autograd mechanism.

"""

import os
from graphviz import Digraph
import numpy as np
import sys
import pathlib 
import math
# from initializer.xavier import Hei , Xavier ,LeCun

def visualize_graph(self, filename='computation_graph'):
        dot = Digraph()
        
        visited = set()

        # Define a recursive function to traverse the graph and add nodes and edges
        def add_node(tensor):
            if tensor not in visited:
                visited.add(tensor)
                if tensor.operation == '':
                    sign = "leaf node"
                else:
                    sign = {tensor.operation}
                dot.node(str(tensor.id), f'{tensor.data}\ngradient: {tensor.grad}\nOperation: {sign}')
                for child in tensor.attached_node:
                    dot.edge(str(tensor.id), str(child.id))
                    add_node(child)
        
        add_node(self)

        dot.render(filename, format='png', cleanup=True)

def toTensor(x:list):
    outcome = []
    # iterating over the list and if a list than recursive calling happen otherwise element will be transformed to Tensor
    for it in x:
        if isinstance(it , list):
            outcome.append(toTensor(it))   # recursive calling in case of list 
        else:
            outcome.append(Tensor(value=it))  # transformation to tensor

    return outcome


def toNumpy(x:list):
    outcome = []
    # iterating over the list and if a list than recursive calling happen otherwise element will be restored from tesnor to a python data tpye
    for it in x:
        if isinstance(it , list):
            outcome.append(toNumpy(x=it))
        else:
            outcome.append(it.data) # restoration from tensor data to normal python data type
    if len(outcome) == 1:
        return outcome[0]    
    return outcome


def dimension_reducer(val:list):
    # traverse until list is nested and 0 item is not get as an primitive data type
    if not isinstance(val, list):
        return ()
    if len(val) == 0:
        return (0,)
    inner_shape = dimension_reducer(val[0])
    return (len(val),) + inner_shape


class Tensor:

    """

        How to approach it :
            01. scalar value will be supported
            02. list or list of list will be considered as valid input and internally handled using numpy
            03. numpy array will also be considered
    
    """

    def __init__(self,value , operation ='', subset = ()) -> None:
        self.data = None
        self.grad = None
        self.operation = operation
        self.attached_node = set(subset)
        self._backward = lambda : None
        self.id = id(self)

        if isinstance(value , (int , float)):

            self.data = value
            self.grad = 0

        elif isinstance(value , list):
            
            self.data = np.array(object= value,dtype=float)
            self.grad = np.zeros_like(a=self.data)

        elif isinstance(value , np.ndarray):

            self.data = value
            self.grad = np.zeros_like(a=self.data,dtype=float)

    
    def __repr__(self) -> str:
        return f"Tensor({self.data},grad :{self.grad})"
    
    def __add__(self , other):
        out = None
        if isinstance(other , (int,float)) and isinstance(self.data , (int , float)):
            other = Tensor(value=other)
            out = Tensor(self.data + other.data,subset=(self, other), operation="Backward<Add>")

        elif isinstance(self.data , np.ndarray) and isinstance(other.data , np.ndarray):
            out = Tensor(value=np.add(self.data , other.data), subset=(self,other),operation="Backward<Add>")
        
        elif isinstance(self.data , np.ndarray) and isinstance(other.data , (int , float)):
            other.data = np.full_like(self.data,fill_value=other.data)
            out = Tensor(self.data + other.data,subset=(self, other),operation="Backward<Add>")

        else:
            out = Tensor(value=self.data+other.data,operation="Backward<Add>",subset=(self, other))
        
        def _backward():
            if isinstance(out.data, (int , float) ):
                self.grad += out.grad * self.data
                other.grad += out.grad * other.data
            else:      
                self.grad += (out.grad * np.ones_like(self.data))
                other.grad += (out.grad * np.ones_like(other.data))

        out._backward = _backward
        return out
    
    def __mul__(self , other):
        out = None
        if isinstance(self.data,(int , float)) and isinstance(other.data , (int , float)):
            out = Tensor(value=self.data*other.data,operation="Backward<Mul>",subset=(self , other))
            def _backward():
                self.grad = out.grad*other.data
                other.grad = out.grad*self.data
            return out

        elif isinstance(self.data , np.ndarray) and isinstance(other.data , (int , float)): # scalar multiplication to a vector or matrix
            out = Tensor(value=self.data*other.data,subset=(self, other),operation="Backward<Mul>")
            def _backward():
                self.grad = out.grad*other.data
                other.grad = out.grad*self.data
            out._backward = _backward
            return out # need to review 
        
        elif isinstance(self.data , np.ndarray) and isinstance(other.data , np.ndarray):
            out = Tensor(value=np.matmul(self.data , other.data),subset=(self,other),operation="Backward<Matmul>")

        def _backward():
            self.grad += np.dot(other.data, out.grad.T).T
            other.grad += np.dot(out.grad.T , self.data).T
        
        out._backward = _backward
        return out
    

    def mean(self):
        out = None
        if isinstance(self.data , np.ndarray):
            out = Tensor(value=np.mean(self.data), subset=(self,),operation="Backward<Mean>")
        def _backward():
            # self.grad = np.full_like(self.data,0.0833)
            self.grad += out.grad / self.data.size
        out._backward = _backward
        return out
    
    def max(self):
        # calculate outcome and setting up backrpop chaining
        outcome = None
        if isinstance(self.data , np.ndarray):
            outcome = Tensor(value=np.max(self.data),subset=(self,),operation="Backward<Max>")
            def _backward():
                self.grad += (self.data==outcome.data) * outcome.grad
            outcome._backward = _backward
    
        
        else:
            raise ValueError("unsupported data type is given!")
        return outcome
    
    def min(self):
        # calculate outcome and setting up the backpropogation chaining through child nodes
        outcome = None
        if isinstance(self.data , np.ndarray):
            outcome = Tensor(value=np.min(self.data),subset=(self,),operation="Backward<Min>")
            def _backward():
                self.grad += (self.data==outcome.data) * outcome.grad
            outcome._backward = _backward
    
        else:
            raise ValueError("unsupported data type is given!")
        return outcome
            
    def log(self):
        # calculate the outcome after taking log with respect to 10 and than setting up backprop chaining with other node 
        outcome = None
        if isinstance(self.data , (int ,float)):
            outcome = Tensor(value=math.log(self.data) , subset=(self,),operation="Backward<Log>")
            def _backward():
                self.grad += outcome.grad * 1/self.data
            outcome._backward = _backward
            
        elif isinstance(self.data , np.ndarray):
            outcome = Tensor(value=np.log(self.data), subset=(self,),operation="Backward<Log>")
            def _backward():

                self.grad += 1/self.data * outcome.grad
            outcome._backward = _backward

        return outcome
    
    def exp(self):
        # calculating the oucome after doing the exponentiation and than setting up the chain for backpropogation with its child nodes
        outcome = None
        if isinstance(self.data ,(int , float)):

            outcome = Tensor(value=math.exp(self.data),subset=(self,),operation="Backward<Exp>")
            def _backward():
                self.grad += outcome.data * outcome.grad
            outcome._backward = _backward
        
        elif isinstance(self.data ,np.ndarray):
            outcome = Tensor(value=np.exp(self.data),operation="Backward<Exp>",subset=(self,))
            def _backward():

                self.grad += outcome.data * outcome.grad
            outcome._backward = _backward
        
        return outcome
    
    def sum(self,axis:int=None):
        outcome = None
        if isinstance(self.data , np.ndarray):
            outcome = Tensor(value=np.sum(self.data , axis=axis),subset=(self,),operation="BackWard<Sum>")
            def _backward():
                if axis:
                    axis_mask = np.expand_dims(np.ones(self.data.shape[axis]), axis=axis)
                    # Distribute gradients to the elements that contributed to the sum
                    self.grad += (axis_mask * outcome.grad)
                else:
                    self.grad +=  outcome.grad * np.ones_like(self.data)
            outcome._backward = _backward

        else:
            raise ValueError("not supported data type")
        
        return outcome
    
    def square(self):
        # implementing the square function for tensor class objects with forward and backward pass also for trashing the computational graph
        outcome = None
        if isinstance(self.data , (int , float)):
            outcome = Tensor(value=self.data * self.data,operation="Backward<Square>",subset=(self,))
            def _backward():
                self.grad += 2*self.data * outcome.grad
            outcome._backward  = _backward

        elif isinstance(self.data , np.ndarray):
            outcome = Tensor(value=np.square(self.data),subset=(self,),operation="Backward<Square>")
            def _backward():
                self.grad += 2*self.data * outcome.grad
            outcome._backward = _backward
    
        else:
            raise ValueError("unsupported data type is encountered!")
        return outcome
    
    def sqrt(self):
        outcome = None
        if isinstance(self.data , np.ndarray):
            outcome = Tensor(value=np.sqrt(self.data),subset=(self,),operation="Backward<Sqrt>")
            def _backward():
                self.grad += (1/(2*outcome.data)) * outcome.grad
            outcome._backward = _backward
        
        elif isinstance(self.data , (int ,float)):
            outcome = Tensor(value=np.sqrt(self.data),subset=(self,),operation="Backward<Sqrt>")
            def _backward():
                self.grad += 1/(2*outcome.data) * outcome.grad
            outcome._backward = _backward

        
        else:
            raise ValueError("unsupported data type is encountered!")
        
        return outcome
    
    def variance(self,axis):
        # implementing the variance forward and backward pass
        outcome = None
        if isinstance(self.data , np.ndarray):
            outcome = Tensor(value= np.var(self.data , axis=axis))
            # backward pass

            def _backward():
                if axis is None:
                    self.grad += outcome.grad * np.ones_like(self.data)

                else:
                    axis_mask = np.expand_dims(np.ones(self.data.shape[axis]), axis=axis)
                    self.grad += (axis_mask * outcome.grad / self.data.shape[axis])
            outcome._backward = _backward

        else:
            raise ValueError("unsupported data type is encountered!")
        
        return outcome

    def std(self,axis:int = None):  # may need to review it testing is compulsory 
        outcome = None
        if isinstance(self.data , np.ndarray):
            out = self.variance(axis=axis)
            outcome = out.sqrt()
            outcome.operation = "Backward<std>"
            outcome.subset = (self,)
            def _backward():
                # Propagate gradients through the square root operation
                out._backward()
                # Propagate gradients through the variance operation (via chain rule)
                self.grad += out.grad / (2 * outcome.data)
            outcome._backward = _backward

        else:
            raise ValueError("unsupported data type is encourted!")
        return outcome
    """
        this is an element wise division so shape must be shape
    
    """
    def __truediv__(self, other):
        outcome = None

        def _backward():  # backpass defining
            if isinstance(other, Tensor):
                self.grad += outcome.grad / other.data
                other.grad -= (outcome.grad * self.data).sum(axis=0)
            else:
                self.grad += outcome.grad / other

        if isinstance(self.data , np.ndarray) and isinstance(other.data , np.ndarray):
            outcome = Tensor(value=self.data / other.data,operation="Backward<Div>",subset=(self, other))
            outcome._backward = _backward


        elif isinstance(self.data , np.ndarray) and isinstance(other.data , (int , float)):
            outcome._backward = _backward

        elif isinstance(self.data , np.ndarray) and isinstance(other, (int , float)):
            other = Tensor(value=other)
            outcome = Tensor(value=self.data / other.data,operation="Backward<Div>",subset=(self, other))
            outcome._backward = _backward

        elif isinstance(self.data ,(int , float)) and isinstance(other , (int , float)):
            other = Tensor(value=other)
            outcome = Tensor(value=self.data / other.data,operation="Backward<Div>",subset=(self, other))
            outcome._backward = _backward

        elif isinstance(self.data , (int , float)) and isinstance(other.data ,(int , float)):
            outcome._backward = _backward

        else:
            raise ValueError("unsupported data type is encountered!")

        return outcome

    def __radd__(self , other):
        return self + other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self , other):
        return other + (-self)

    def __rtruediv__(self, other): 
        return self / other

    def backward(self):  # this function is the base function which generate computational graph for each node or tensor
        graph_nodes = []
        visited = set()
        def build_graph(v):
            if v not in visited:
                visited.add(v)
                for child in v.attached_node:
                    build_graph(child)
                graph_nodes.append(v)
        build_graph(self)
        if isinstance(self.data, (int , float)):
            self.grad = 1
        else:
            self.grad = np.ones_like(self.data)
        # print(graph_nodes)
        for v in reversed(graph_nodes):
            v._backward()



if __name__ == '__main__':
    print('let us start!')
    # taking an example and verified using pytorch autograd 
    arr1 = [
        [1.2 , 1.3, 1.4],
        [2.3, 4.5, 5.6],
        [12.3, 14.5, 15.6]
    ]

    arr2 = [
        [7.2 , 1.3, 1.4,5.6],
        [2.3, 13.4, 5.6,7.9],
        [8.1, 6.1, 5.5,8.1]
    ]

    arr3 = [
        [1.2 , 1.3, 13.4,2.3],
        [7.3, 2.5, 5.6,6.7],
        [4.1, 5.1, 12.5,5.6],
        [11.2,23.2,1.3,2.4]
    ]

    arr4 = [
        [1.7 , 1.13, 7.4,4.5],
        [2.5, 4.8, 9.6,5.5],
        [3.0, 4.8, 8.5,6.5]
    ]


    # obj1 = Tensor(value=arr1)
    # obj2 = Tensor(value=arr2)
    # obj3 = Tensor(value=arr3)
    # obj4 = Tensor(value=arr4)

    # out1 = obj1 * obj2
    # out2 = obj4 * obj3

    # out = out1 + out2
    # loss = out.mean()
    # loss.backward()
    # visualize_graph(self=loss,filename = 'img/custom_autograd')
    # print('renadered!')

    maxi = Tensor(value=np.random.random(size=(10 , 10)))
    mini = Tensor(value=np.random.random(size= (10 , 10)))
    val = maxi / mini
    print(val)



