"""
    this class implement pooling layer for 
    3 color channels rgb image data.

"""
from Tensor.tensor import Tensor
import os
import sys
import pathlib
import numpy as np

def average_finder(arr):
    return np.average(arr)


def max_finder(arr):
    return np.max(arr)

def min_finder(arr):
    return np.min(arr)


def template_func(batch , row , col,color_channel, padding , stride , kernel_size, value_list , func_name,x):
    padded_x = np.pad(x.data, ((0, 0), (padding,padding), (padding, padding),(0,0)), mode='constant')

    output_rows = int(np.ceil((row + 2 * padding - kernel_size  - 1) / stride) + 1)
    output_cols = int(np.ceil((col + 2 * padding - kernel_size - 1) / stride) + 1)

    for i in range(0, batch):
        for ch in range(0, color_channel):
            for j in range(0, row):
                if j + kernel_size > row + 2 * padding:
                    break
                for k in range(0, col):
                    if k + kernel_size > col + 2 * padding:
                        break
                    if len(value_list) == batch *color_channel* output_cols * output_rows:
                        break
                    filter_mat = padded_x [i, j:j +kernel_size, k:k + kernel_size,ch]
                    # print(filter_mat)
                    if func_name == "max":
                        val = np.max(filter_mat)
                    elif func_name == "min":
                        val = np.min(filter_mat)
                    elif func_name == "avg":
                        val = np.average(filter_mat)
                    else:
                        raise NameError("unsupported pooling method!")
                    value_list.append(val)
                    k += kernel_size

                if len(value_list) == batch * output_cols * output_rows*color_channel:
                    break
                j += kernel_size
            if len(value_list) == batch * output_cols * output_rows*color_channel :
                    break

    return Tensor(value=np.array(value_list).reshape(batch,color_channel, output_rows, output_cols))

class MaxPool3D:

    def __init__(self, kernal_size = 3, padding = 1, stride = 2):
        self.kernal_size = kernal_size
        self.padding = padding
        self.stride = stride
        self.value_list = []


    def maxpool(self , x:Tensor):

        if len(x.data.shape ) == 4:
            batch , color_channel , row , col = x.data.shape
        else:
            raise ValueError("not supported image data!")
        
        return template_func(batch=batch , row=row , col= col ,color_channel=color_channel, stride=self.stride , padding = self.padding, kernel_size=self.kernal_size,func_name="max",value_list=self.value_list,x=x)
        

class MinPool3D:

    def __init__(self, kernal_size = 3, padding = 1, stride = 2):
        self.kernal_size = kernal_size
        self.padding = padding
        self.stride = stride
        self.value_list = []


    def minpool(self , x:Tensor):

        if len(x.data.shape ) == 4:
            batch , color_channel , row , col = x.data.shape
        else:
            raise ValueError("not supported image data!")
        
        return template_func(batch=batch , row=row , col= col ,color_channel=color_channel, stride=self.stride , padding = self.padding, kernel_size=self.kernal_size,func_name="min",value_list=self.value_list,x=x)
        


class AveragePool3D:

    def __init__(self, kernal_size = 3, padding = 1, stride = 2):
        self.kernal_size = kernal_size
        self.padding = padding
        self.stride = stride
        self.value_list = []


    def maxpool(self , x:Tensor):

        if len(x.data.shape ) == 4:
            batch , color_channel , row , col = x.data.shape
        else:
            raise ValueError("not supported image data!")
        
        return template_func(batch=batch , row=row , col= col,color_channel=color_channel , stride=self.stride , padding = self.padding, kernel_size=self.kernal_size,func_name="avg",value_list=self.value_list,x=x)
        


