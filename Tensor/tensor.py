"""
    this is a base class for implementing all the nueral network.
    All the neural network will implemented on the top of it.
"""
import math
from utility import visualize_graph as vz

class Tensor():

    def __init__(self,value,subset = (),operation = '') -> None:
        self.data = value
        self.grad = 0
        self._backward = lambda : None
        self.children = set(subset)
        self.sign = operation
        self.id = id(self)

    def __repr__(self) -> str:
        return f"Tensor({self.data})"

    def __add__(self,other):
        # there will be four case 2 + 3 , 2 + Tensor , 4 + Tensor , Tensor + Tensor
        if isinstance(other,Tensor):
            out = Tensor(self.data+other.data,subset=(self,other),operation='+')
        else:
            other = Tensor(other)
            out = Tensor(self.data+other.data,operation='+',subset=(self,other))

        def _backward():

            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out
    
    def __radd__(self,other):
        return self + other
    

    def __mul__(self, other):
        # it will also be in four ways : 2 * 3 , Tensor * 2, 2 * Tensor, Tensor * Tensor
        if isinstance(other,Tensor):
            out = Tensor(self.data*other.data,operation='*',subset=(self,other))
        else:
            other = Tensor(other)
            out = Tensor(other.data*self.data,operation='*',subset=(self,other))

        def _backward():
            self.grad += other.data*out.grad  # chain rule 
            other.grad += self.data*out.grad  # chain rule
        
        out._backward = _backward
        return out
    
    def __rmul__(self,other):
        return self*other
    
    
    def __neg__(self):
        return self*-1

    def __sub__(self,other):
        return self + (-other)
    
    def __rsub__(self,other):
        return other + (-self)
    

    def __pow__(self,other):
        out = Tensor(self.data**other,subset=(self),operation=f"**{other}")
        
        def _backward():
            self.grad += (other*self.data**(other-1)) * out.grad

        out._backward = _backward
        return out
    
    def __truediv__(self,other):
        return self * other**-1
    
    def __rtruediv__(self,other):
        return other * self**-1
    
    def exp(self):
        out = Tensor(math.exp(self.data),operation="exp",subset=(self,))
        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out
    
    def log10(self):
        out = Tensor(math.log10(self.data),operation="log10",subset=(self,))
        def _backward():
            self.grad += (1/self.data) * out.grad

        out._backward  = _backward
        return out
    
    """
        there are a lot of activation function that are used in deep learning.
        Activation function that are implemented here are :
            01. Relu
            02. Cross Entropy
            03. Sigmoid
            04. Leaky Relu
            05. Selu
            06. Gelu
            07. elu
            08. tanh
            09. softplus
            10. swish

    """
    def tanh(self):
        nominator = math.exp(2*self.data) - 1
        denominator = math.exp(2*self.data) + 1
        if denominator != 0:
            out = Tensor(nominator/denominator,subset=(self,),operation='tanh')
        else:
            h = 0.00001
            out = Tensor(nominator/(denominator+h),subset=(self,),operation='tanh')

        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward
        return out
        

    def relu(self):
        if self.data > 0:
            out = Tensor(self.data,operation="relu",subset=(self,))
        else:
            out = Tensor(0,subset=(self,),operation="relu")

        def _backward():
            if self.data > 0:
                self.grad += 1 * out.grad
            else:
                self.grad += 0
        out._backward = _backward
        
        return out
    

    def leaky_relu(self,lamda= 0.01):
        if self.data > 0:
            out = Tensor(self.data,operation="leaky_relu",subset=(self,))
        else:
            out = Tensor(lamda*self.data,operation="leaky_relu",subset=(self,))
        
        def _backward(self):
            if self.data > 0:
                self.grad += out.grad * 1
            else:
                self.grad += lamda * out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self):
        # e^x/ 1 + e^x
        a = math.exp(self.data)
        out = Tensor(a/(1+a) , operation="sigmoid", subset=(self,))

        def _backward():
            self.grad += (out.data * (1- out.data)) * out.grad

        out._backward = _backward
        return out
    
    def elu(self,alpha = 0.01):

        if self.data > 0:
            out = Tensor(self.data,operation="elu",subset=(self,))
        else:
            out = Tensor(alpha*(math.exp(self.data) - 1),operation="elu",subset=(self,))

        def _backward():
            if self.data >0:
                self.grad += out.grad * 1
            else:
                self.grad += alpha*math.exp(self.data)*out.grad
        out._backward = _backward
        return out
    
    def swish(self,beta = 0.001):
        # f(x) = x * sigmoid(beta*x)
        a = math.exp(beta*self.data)
        reuseable_value = a/(1+a)
        out = Tensor(self.data*reuseable_value,operation="swish",subset=(self,))
        def _backward():
            self.grad += ((reuseable_value) + (self.data * 1/beta * out.data(1-out.data)))*out.grad
        return out

    def gelu(self):
        # f(x) = x * phi(x) 
        # where phi(x) is normal probability distribution
        # phi(x) = 1/(2*pi)**0.5 * e-x**2/2
        val = math.sqrt(2*3.14) * math.exp(-self.data**2/2)
        out = Tensor(self.data*val,operation="gelu",subset=(self,))
        def _backward():
            self.grad += ( (val) + (self.data*val*-1)  )*out.grad
        out._backward = _backward
        return out

    def selu(self,lamda=0.001,alpha=0.001):
        # f(x) = lamda * {
        #                   if x > 0 : x
        #                   else: alpha*e^e - alpha
        #           }
        if self.data>0:
            out = Tensor(lamda*self.data,operation="selu",subset=(self,))
        else:
            out = Tensor(lamda* (alpha*math.exp(self.data) - alpha),operation="selu",subset=(self,))

        def _backward():
            if self.data>0:
                self.grad += lamda*out.grad
            else:
                self.grad += lamda * (alpha*math.exp(self.data)) * out.grad
        out._backward = _backward
        return out
    
    def softplus(self):
        # f(x) = log(1+e^x)
        pow_self = math.exp(self.data)
        out = Tensor(math.log10(1+pow_self))
        def _backward():
            self.grad += ((1/(1+pow_self)) + pow_self)*out.grad
        out._backward = _backward
        return out
    

    def backward(self):
        graph_nodes = []
        visited = set()
        def build_graph(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_graph(child)
                graph_nodes.append(v)
        build_graph(self)
        self.grad = 1
        # print(graph_nodes)
        for v in reversed(graph_nodes):
            v._backward()



if __name__ == '__main__':
    a = Tensor(10)
    b = Tensor(20)
    c = Tensor(30)
    d = Tensor(5)

    e = a * b + c
    f = 2 * e
    f.backward()
    # print(c.grad)

    vz(self=e)