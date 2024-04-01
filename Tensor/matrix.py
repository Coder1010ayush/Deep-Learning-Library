from graphviz import Digraph
import numpy as np
import math
import os
import sys
class Tensor:
    """"
    
        This is a class that implements a general Tensor class building block for nueral networks.
        this class supported a single scalar value and matrix value also.
        this class internally make a computational graph for calculating gradient and there is a 
        backward() function that allows to backpropogate through the graph and change the gradient of 
        an expression.

        This class also implements a number of activation function with defining its foward propogation 
        as well as backward propogation simultaneousaly.

        Note : There is a possibility to make a different class for all the activation function so that code readability and 
        solid principle of software propgramming will not be voilating. 

        Todo : Organise code according to solid prinicple.  #important !!@@!!
        
    """
    def __init__(self,value, subset = (), operation= "") -> None:
        if isinstance(value, (int,float)):
            # this means value is scalar value 
            self.data = np.array(value,dtype=float)
        else:
            # this means value is list 
            self.data = np.array(object=value,dtype=float)
        self.sign = operation
        self.grad = np.full_like(self.data,0.001)
        self._backward = lambda : None
        self.children = set(subset)
        self.id = id(self)

    def backward(self):  # this function is the base function which generate computational graph for each node or tensor
        graph_nodes = []
        visited = set()
        def build_graph(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_graph(child)
                graph_nodes.append(v)
        build_graph(self)
        self.grad = np.ones_like(self.data)
        # print(graph_nodes)
        for v in reversed(graph_nodes):
            v._backward()

    def shape(self):
        # print( f"TensorSize{(self.data).shape}")
        return (self.data).shape

    def __repr__(self) -> str:
        return f"Tensor(data:{self.data},grad:{self.grad})"

    def __add__(self,other):
        if isinstance(other, (int,float)):
            other = Tensor(value=other)
            out = Tensor(np.add(self.data,other.data),subset=(self,other),operation="add")

        # elif isinstance(other, list):
        #     if isinstance(self, Tensor):
        #         ls = []
        #         for   item in other:
        #             ls.append(Tensor(value=self.data + item.data))
        #     return ls   
        
        else:
            out = Tensor(np.add(self.data,other.data),subset=(self,other),operation="add")

        def _backward():
            self.grad += out.grad * np.ones_like(self.data)
            other.grad += out.grad * np.ones_like(other.data)

        out._backward = _backward
        return out
    
    def __radd__(self,other):
        return self + other
    
    def __mul__(self,other):
        if isinstance(other,(int,float)):
            other = Tensor(value=other)
            out = Tensor(value=self.data * other.data,subset=(self,other),operation="mul")
       
        else:
            out = Tensor(value=np.dot(self.data,other.data),subset=(self, other),operation="mul")

        def _backward():
            self.grad += np.dot(out.grad, other.data.T) # chain rule
            other.grad += np.dot(self.data.T ,out.grad) # chain rule

        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * -1

    def __sub__(self , other ):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __pow__(self , other ):
        # this will increase all the elements for the tensor to power of other 
        # here other is scalar value 
        if isinstance(other, (int,float)):
            out = Tensor(value=self.data**other,subset=(self,),operation=f"**{other}")

        elif isinstance(self, list):
            if isinstance(other, Tensor):
                ls = []
                for   item in self:
                    ls.append(Tensor(value=item.data ** other.data))
            return ls  
         
        else:
            raise ValueError("Not supported now")
            
        def _backward():
            self.grad += other * self.data**(other-1) *out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self,other):
        if isinstance(other , (int ,float)):
            return self * (other**-1)
        else:
            raise ValueError("right now not supported!")
        
    def __rtruediv__(self,other):
        return other * (self**-1)


    def visualize_graph(self, filename='computation_graph'):
        dot = Digraph()
        
        visited = set()

        # Define a recursive function to traverse the graph and add nodes and edges
        def add_node(tensor):
            if tensor not in visited:
                visited.add(tensor)
                if tensor.sign == '':
                    sign = "leaf node"
                else:
                    sign = {tensor.sign}
                dot.node(str(tensor.id), f'{tensor.data}\ngradient: {tensor.grad}\nOperation: {sign}')
                for child in tensor.children:
                    dot.edge(str(tensor.id), str(child.id))
                    add_node(child)
        
        add_node(self)

        dot.render(filename, format='png', cleanup=True)
        print("rendered")


    """
        implementing backpropogation for all the activation functions that are commonly used : 
            -a.  Relu
            -b.  Leaky_relu
            -c.  selu
            -d.  gelu
            -e.  sigmoid
            -f.  tanh
            -g.  swish
            -h.  softplus
            -i.  elu  
            -j.  softmax / cross entropy
        # more will be added in future
    """

    def tanh(self):     # implementing tanh function - backpropogation
        """
        function formulae : {
            f(x) = (e^x - e^-x)/(e^x + e^-x)
            differentiation with respect to x :
            f'(x) = 1 - tanh(x) ** 2
        }
        """
        outcome = Tensor(value=np.tanh(self.data),subset=(self,),operation="tanh")
        def _backward():
            self.grad += (1-outcome.data**2)*outcome.grad
        outcome._backward = _backward
        return outcome
    
    
    def relu(self):  # implementing backpropogation for relu function 
        """
        function formulae : {
            f(x) = {
                    x, x>=0
                    0, x<0
            }
        }
        
        """
        def filter(x):
            return np.maximum(0,x)
        
        outcome = Tensor(value=filter(self.data),subset=(self,),operation="relu")
        result = np.where(self.data > 0, 1, 0)
        def _backward():
            self.grad = result * outcome.grad
        outcome._backward = _backward
        return outcome
    

    def leaky_relu(self,lamda=0.001):  # implementing backpropogation for leaky_relu function
        """
        function formula: 
            f(x)  = {
                x : x >= 0
                lamda : x < 0 
            }
        """
        def filter(x):
            return np.maximum(0,x)
        outcome = Tensor(value=np.where(self.data>0,filter(self.data),lamda),subset=(self,),operation="leaky_relu")
        result = np.where(self.data>0,1,lamda)
        def _backward():
            self.grad = result * outcome.grad

        outcome._backward = _backward
        return outcome
    
    def sigmoid(self):  # implementing backpropogation for sigmoid function for a numpy array
        """
        function formula : 
            f(x) = 1/(1+e^-x)
            diffentiation of f(x) with respect to x = 
                f'(x) = f(x) * (1-f(x))
        """
        def filter(x):
            return (1/(1+np.exp(x)))
        
        outcome = Tensor(value=filter(self.data),subset=(self,),operation="sigmoid")
        def _backward():
            self.grad = (outcome.data * (1-outcome.data)) * outcome.grad

        outcome._backward = _backward
        return outcome

    def swish(self,beta= 0.001):  # implementing swish activation function's backpropogation for whole graph
        """ 
            function formulae : 
                f(x) : x * sigmoid(beta*x)
            differentiation of f(x) with respect to x = 
                f'(x) = x (sigmoid(beta*x)-(1-sigmoid(beta*x) ) + 1 * sigmoid (beta*x)
        """
        def filter(x,beta):
            return (1/(1+np.exp(beta*x)))
        outcome = Tensor(self.data*filter(self.data,beta=beta),subset=(self,),operation="swish")
        def _backward():
            self.grad = ( (self.data)*(filter(self.data,beta=beta) *(1-filter(self.data,beta=beta)) ) + filter(self.data,beta=beta)  )*outcome.grad

        outcome._backward = _backward
        return outcome
    

    def elu(self,alpha=0.001):  # implementing backpropogation for elu activation function 
        """
        function formulae :
                R(z) = {
                    z ,z>0
                    alpha.(e^z - 1) , z<=0
                }
        differentiation of f(x) with respect to x = 
                R'(z) = {
                    1  , z>0
                    alpha.e^z<0
                    
                }
        """
        def filter(x):
            return np.where(x>0,x,alpha*(np.exp(x)-1))

        outcome = Tensor(value=filter(self.data),subset=(self,),operation="elu")
        def _backward():
            self.grad  = alpha*np.exp(self.data) * outcome.grad
        outcome._backward = _backward
        return outcome
    
    def gelu(self):  # implementing backpropogation for gelu activation function
        """
        function formulea:
            f(x) = phi(x) * x   [ where phi(x) = 1/(2*pi)**0.5 * e-x**2/2 ]
                where phi(x) is the function of normal probabilistic distribution 
            differentiation with respect to x = 
            f'(x) = phi'(x)*x + 1*phi(x)
        """
        val = np.sqrt(2*3.14) * np.exp(-self.data**2/2)
        outcome = Tensor(value=self.data*val , subset=(self,),operation="gelu")
        
        def _backward():
            self.grad = ( (val) + (self.data*val*-1)  )*outcome.grad
        outcome._backward = _backward
        return outcome
    

    def selu(self,alpha = 0.001):    # implementing backpropogation for selu function -- not completed testes may be modified in future 
        # f(x) = lamda * {
        #                   if x > 0 : x
        #                   else: alpha*e^x - alpha
        # }
        def filter(x,alpha):
            return np.where(x>0,x,alpha*np.exp(x)-alpha)
        
        outcome = Tensor(value=filter(self.data,alpha=alpha),subset=(self,),operation="selu")
        result = np.where(self.data>0,1,alpha*np.exp(self.data))
        def _backward():
            self.grad = result*outcome.grad
        outcome._backward = _backward
        return outcome
    
    """"
        # todo : implement below activation function
    """
    
    # def binary_cross_entropy(self, other):  # here other  represents the target vector 
    #     """"
    #         function for binary cross entropy is given below : 
    #         f(y{i}) = -1/N ( sigma (i=1 to n) y{i} * log(y{i}  (1-y{i})*log(1-y{i})) )
        
    #     """
    #     eps = 1e-7  # Small value to avoid numerical instability
    #     loss_value = -np.mean(other.data * np.log(self.data + eps) + (1 - other.data) * np.log(1 - self.data + eps))
    #     loss_tensor = Tensor(value=loss_value, subset=(self, other), operation='binary_cross_entropy')

    #     # defining backward pass or backpropogation for the binary_cross_entropy activation function 
    #     def _backward():
    #         self.grad += (self.data - other.data) / (self.data * (1 - self.data) + eps) * loss_tensor.grad

    #     loss_tensor._backward = _backward
    #     return loss_tensor
    
    # def categorical_cross_entropy(self , other):
    #     """ 
    #         function of categorical cross entropy is given below :
    #         f =  {
    #                     sigma(i=1 to n) sigma(j=1 to c) y{i}{j}' log(y{i}{j}')
    #                 }
    #     """
    #     eps = 1e-7  # Small value to avoid numerical instability
    #     loss_value = -np.mean(np.sum(other.data * np.log(self.data + eps), axis=1))
    #     loss_tensor = Tensor(value=loss_value, subset=(self, other), operation='categorical_cross_entropy')

    #     def _backward():
    #         self.grad += -other.data / (self.data + eps) * loss_tensor.grad

    #     loss_tensor._backward = _backward
    #     return loss_tensor
    

    #  # implementation of forward and backward propoogation of mean square error activation function  - generally used for linear or logistic regression or regression problems
    #     shape = self.data.shape
    #     N = 1
    #     for i in shape:
    #         N *= i
    #     mse_value = np.mean(self.data - other.data)
    #     mse_tensor = Tensor(value=mse_value, subset=(self, other), operation='mse')
    #     def _backward():
    #         self.grad = 2/N * (self.data - other.data) * mse_tensor.grad
    #     mse_tensor._backward = _backward
    #     return mse_tensor
    
    def mse( self , other ): # implementation of forward and backpropogation of mean square activation function - generally used for linear or logistic regrressio or regression problems
        # note that here self is y_pred , other is y_actual
        # loss is calculated as the element wise difference of self and other and than taking the mean of that vector
        N = self.data.flatten().shape[0]
        loss = Tensor(value=np.mean( np.square ( self.data - other.data )  ),operation="mse",subset=(self,other) )
        def _backward():
            self.grad = (2 * (self.data - other.data) ) / N
            other.grad = (2 * (other.data - self.data)) / N
        loss._backward = _backward
        return loss
    
    def rmse(self , other):  # implementation of forward and backpropogation of root mean squared error loss function - generally used for regression problems where outcome would be a numerical values.
        
         # note that here self is y_pred , other is y_actual
         # loss is calculated as the element wise difference of self and other and than taking the mean of that vector and than square root of it.
         # this is similar to mse loss function mainly used for numerical value prediction or regression problems in linear regression and logistics regression like problems 
        shape = self.data.shape
        N = 1
        for i in shape:
            N *= i
        loss_value = np.sqrt(np.mean(self.data - other.data))
        mse_loss = np.mean(np.square(self.data - other.data))
        loss = Tensor(value=loss_value , subset=(self,other),operation="rmse")
        def _backward():

            self.grad += (1/2*mse_loss) * ((2*(self.data - other.data))/N) * loss.grad
            other.grad += (1/2*mse_loss) * ((2*(other.data - self.data))/N) * loss.grad
        loss._backward = _backward
        return loss
            

    def softmax(self, other):   # implementation of forward and backward propogation of softmax function for multi class labeling problems 
        # this is like a probablistic approach to find out the loss given y_pred and y_actual
        
        exp_values = np.exp(self.data)
        softmax_values = other.data / exp_values
        outcome = Tensor(value=softmax_values, subset=(self,), operation='softmax')

        def _backward():
            jacobian_matrix = np.diag(softmax_values) - np.outer(softmax_values, softmax_values)
            self.grad = np.dot(jacobian_matrix, outcome.grad.T)

        outcome._backward = _backward
        return outcome
        
 
    def hubour(self , other , alpha:float): # implemention of forward and backward propogation of hubiur loss function
        """
            this is a loss function which is used for maily regression (numerical output like problems).
            it lies in between of mse (mean squared loss function) and rmse (root mean square loss function)
            Mathematical function is :
                f = {
                    1/2 (y_pred - y_actual ) **2 if |(y_pred-y_actual)| < alpha
                    alpha * (y_pres-y_actual ) - 1/2 * alpha * alpha , where |y_pred-y_actual| > alpha
                }
        """

        # alpha would be given
        error = np.abs(self.data - other.data)
        error_loss = np.mean(error)
        if error_loss <= alpha:
            outcome = Tensor(value=1/2 *np.square(self.data - other.data) , subset=(self, other), operation="hubour" )
        else:
            outcome = Tensor(value=alpha * (self.data - other.data) -( 1/2*alpha ) , subset=(self,other), operation="hubour" )


        def _backward():

            if alpha >= error_loss:
                self.grad = 1 * error * outcome.grad
                other.grad = error * outcome.grad 
            else:
                self.grad = alpha*other.data * outcome.grad
                other.grad = 0

        outcome._backward = _backward
        return outcome



    def binary_cross_entropy(self, other):  # implementation of forward and backward pass of binary cross entropy
        
        """
            this loss function is applicable for binary classes labels data.
            it is also known as log loss activation function.
            it is a standard activation function that is used in case of binary class classification.
        
            Mathematical formulation is given below :
                L( y_true , y_pred ) = -[ y_true * log(y_pred)  + (1-y_true) * log( 1 - y_pred ) ]
            
        """
        #let us assume that self is y_true and other is y_pred
        if self.data.shape == other.data.shape:
            N = 1
            for i in self.data.shape:
                N  *= i

            outcome = Tensor(value=-1*(self.data * np.log(other.data)+ ((1-self.data) * np.log(1-other.data)) ) )

            def _backward():
                self.grad = np.log(1-other.data) - np.log(other.data)
                other.grad = 1/(1-other.data) - self.data/other.data  - self.data/(1-other.data)


            outcome._backward = _backward
            return outcome
        else:
            raise ValueError(f"shape is not compatible {self.data.shape} and {other.data.shape}")
        

        # todo
    def categorical_cross_entropy(self,other):   # implementation of forward and backward pass of categorical cross entropy
        pass


if __name__ == '__main__':
    x = Tensor(value=[1,-2,3])
    y = Tensor(value= [1,2,3])
    p = x + y
    a = Tensor(value=[1,2,3])
    b = Tensor(value= [1,2,3])
    q = a * b

    z = p*q
    out = z.selu()
    print("out : ",out)
    out.backward()
    print("a grad : ", a.grad)
    print("b grad : ", b.grad)

    print()
    print()

    print("x grad : ", x.grad)
    print("y grad : ", y.grad)

    print()
    print()

    print("z data : ",z)
    print("z grad : ",z.grad)
    print("out grad : ",out.grad)

    out.visualize_graph()

    arr1 = np.array([1,2,3])
    arr2 = np.array([2,3,4])
    print(arr1*arr2)

    