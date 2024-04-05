"""
    this class implements cnn architecture.
    Some 
"""
import os
import sys
from Tensor.tensor import Tensor
from Tensor.utility import visualize_graph
import torch
from torch import nn
from torch import functional as F
import numpy as np
from scipy import signal

""""
    this class implement a general cnn class.
    Pros : handling in_channel in single class  not made conv1d , conv2d and conv3d classifier
    Cons : does not support stride and padding.

"""


class conv:


    def __init__(self,image_shape , kernal_size , number_of_layers_of_kernal) -> None:
        self.image_shape = image_shape
        self.image_depth , self.image_heigth , self.image_width = image_shape
        self.kernel_layers = number_of_layers_of_kernal
        self.kernal_size = kernal_size
        self.output_shape = (self.kernel_layers, self.image_heigth - self.kernal_size + 1, self.image_width - self.kernal_size + 1)
        self.kernels_shape = (self.kernel_layers, self.image_depth, self.kernal_size, self.kernal_size)
        self.kernel = np.random.random(size=self.kernels_shape)
        self.bias = np.random.random(size=self.output_shape)
        self.grad = 0

    def __repr__(self) -> str:
        string = f"Conv(\n('kernal_size': {self.kernal_size}),\n('kernal_layer : {self.kernel_layers}),\n('input_shape' : {self.image_shape})\n)"
        return string


    def convolve(self,x):
        self.x = x.data
        self.output = np.copy(self.bias)

        for cnt in range(0,self.kernel_layers):

            for it in range(0, self.image_depth):
                self.output[cnt] += signal.correlate2d(self.x[it], self.kernel[cnt, it], "valid") # convolve operation using scipy 

        return self.output


    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        grad = np.zeros(self.image_shape)

        for i in range(self.kernel_layers):
            for j in range(self.image_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.x[j], output_gradient[i], "valid")
                grad[j] += signal.convolve2d(output_gradient[i], self.kernel[i, j], "full")

        self.kernel -= learning_rate * kernels_gradient
        self.bias -= learning_rate * output_gradient
        self.grad = grad
        return grad


        

import numpy as np
from scipy import signal


"""
    below class implements also stride and padding but there is an issue with backpropogation code 
    may be updated in future.

"""

class Conv:
    def __init__(self, image_shape, kernel_size, number_of_kernels, stride=1, padding=0):
        self.image_shape = image_shape
        self.image_depth, self.image_height, self.image_width = image_shape
        self.kernel_layers = number_of_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Calculating output shape
        self.output_height = (self.image_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        self.output_width = (self.image_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        self.output_shape = (self.kernel_layers, self.output_height, self.output_width)
        
        # Initializing kernels and bias
        self.kernels_shape = (self.kernel_layers, self.image_depth, self.kernel_size, self.kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.bias = np.random.randn(*self.output_shape)

    def __repr__(self):
        return f"Conv(\n\tkernel_size={self.kernel_size},\n\tkernel_layers={self.kernel_layers},\n\tinput_shape={self.image_shape},\n\tstride={self.stride},\n\tpadding={self.padding}\n)"

    def convolve(self, x):
        self.x = x
        self.output = np.copy(self.bias)

        # kuchh bhi
        x_padded = np.pad(self.x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        # Perform convolution
        for k in range(self.kernel_layers):
            for d in range(self.image_depth):
                for i in range(0, self.output_height):
                    for j in range(0, self.output_width):
                        self.output[k, i, j] += np.sum(
                            x_padded[d,
                                     i * self.stride : i * self.stride + self.kernel_size,
                                     j * self.stride : j * self.stride + self.kernel_size] *
                            self.kernels[k, d]
                        )

        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros_like(self.x)  # Initialize gradient for input

        # Add padding to the output gradient for convolution with the kernel
        output_gradient_padded = np.pad(
            output_gradient,
            ((0, 0), (self.kernel_size - 1, self.kernel_size - 1), (self.kernel_size - 1, self.kernel_size - 1)),
            mode='constant'
        )

        # Perform backpropagation
        for k in range(self.kernel_layers):
            for d in range(self.image_depth):
                for i in range(self.output_height):
                    for j in range(self.output_width):
                        # Extract the patch from the padded output gradient
                        patch = output_gradient_padded[k,
                                                    i * self.stride : i * self.stride + self.kernel_size,
                                                    j * self.stride : j * self.stride + self.kernel_size]
                        # Compute the gradient for the kernel
                        kernels_gradient[k, d] += signal.correlate2d(self.x[d], patch, mode='valid')
                        # Compute the gradient for the input with the kernel (not flipped)
                        input_gradient[d,
                                    i * self.stride : i * self.stride + self.kernel_size,
                                    j * self.stride : j * self.stride + self.kernel_size] += signal.convolve2d(patch, self.kernels[k, d], mode='full')

        # Remove padding from the input gradient if needed
        if self.padding > 0:
            input_gradient = input_gradient[:, self.padding:-self.padding, self.padding:-self.padding]

        # Update kernels and bias
        self.kernels -= learning_rate * kernels_gradient
        self.bias -= learning_rate * output_gradient
        return 
    

if __name__ == '__main__':
    print('let us start!')