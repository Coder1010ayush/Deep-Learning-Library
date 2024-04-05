"""
    in this file the majorely used function will be implemented for Tensor class
    such :
        matrix multiplication
        matrix transformation
        matrix splitting 
        matrix dot product
        vectorizer class

"""
import os
import sys
import pathlib
from Tensor.tensor import Tensor
import math
import numpy as np



class Operations(Tensor):

    def matmul(f_tensor:Tensor , s_tensor:Tensor):

        """
            this function multiply two tensor 
        """
        outcome = np.multiply()
