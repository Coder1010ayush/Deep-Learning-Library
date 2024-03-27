"""
    this class will implemented all kinds of 
    pooling layer such as :

        01. Max Pooling layer
        02. Average Pooling layer
        03. Min Pooling layer 

"""
import numpy as np
import scipy as sc
import os
import sys
from Tensor.matrix import Tensor
from Tensor.utility import visualize_graph

class MaxPool:

    def __init__(self,kernel_size = 3,stride = 2 ,padding = 1) -> None:
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.value_list = list()
        
    def maxpool(self, x):
        if len(x.data.shape) == 3:
            batch, row, col = x.data.shape
        elif len(x.data.shape) == 2:
            row, col = x.data.shape
            batch = 1
            x.data = np.expand_dims(x.data, axis=0)  
        else:
            raise ValueError("Not supported!")

        padded_x = np.pad(x.data, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        output_rows = int(np.ceil((row + 2 * self.padding - self.kernel_size  - 1) / self.stride) + 1)
        output_cols = int(np.ceil((col + 2 * self.padding - self.kernel_size - 1) / self.stride) + 1)

        for i in range(0, batch):
            for j in range(0, row):
                if j + self.kernel_size > row + 2 * self.padding:
                    break
                for k in range(0, col):
                    if k + self.kernel_size > col + 2 * self.padding:
                        break
                    if len(self.value_list) == batch * output_cols * output_rows:
                        break
                    filter_mat = padded_x[i, j:j + self.kernel_size, k:k + self.kernel_size]
                    val = np.max(filter_mat)
                    self.value_list.append(val)
                    k += self.kernel_size

                if len(self.value_list) == batch * output_cols * output_rows:
                    break
                j += self.kernel_size
            if len(self.value_list) == batch * output_cols * output_rows :
                    break

        return Tensor(value=np.array(self.value_list).reshape(batch, output_rows, output_cols))

class AveragePool:

    def __init__(self , kernal_size = 3, padding = 1, stride = 1) -> None:
        self.kernal_size = kernal_size
        self.padding = padding
        self.stride = stride
        self.value_list = []


    def avgpool(self,x):
        if len(x.data.shape) == 3:
            batch , row , col = x.data.shape
            

        elif len(x.data.shape) == 2:
            row , col = x.data.shape
            batch = 1

        else:
            raise ValueError("not supported shape of image_data!")
        

        padded_arr = np.pad(x.data , ((0,0), (self.padding,self.padding),(self.padding  , self.padding) ), mode="constant")
        out_cols = int(np.ceil( (row+2*self.padding - self.kernal_size - 1) /self.stride )+1)
        out_rows = int(np.ceil( (col + 2*self.padding - self.kernal_size - 1)/ self.stride) + 1)

        for i in range(0 , batch):

            for j in range(0 , row):
                if j + self.kernal_size > row + 2*self.padding:
                    break
                for k in range(0 , col):
                    if k + self.kernal_size > col + 2*self.padding:
                        break

                    filter_map = padded_arr[i, j:j+self.kernal_size , k:k+self.kernal_size]
                    val = np.average(filter_map)
                    # print(val)
                    self.value_list.append(val)
                    K = k+self.stride
                if len(self.value_list) == batch * out_cols * out_rows :
                    break

                j = j+ self.kernal_size
            if len(self.value_list) == batch * out_cols * out_rows :
                    break
        # print(batch * out_cols * out_rows)
        return Tensor(value=np.array(self.value_list).reshape(batch , out_rows , out_cols))


class MinPool:
    def __init__(self, kernal_size = 3 , padding = 1 , stride = 1) -> None:
        self.value_list = []
        self.kernal_size = kernal_size
        self.stride = stride
        self.padding = padding

    def minpool(self , x):
        if len(x.data.shape) == 3:
            batch , row , col = x.data.shape
            

        elif len(x.data.shape) == 2:
            row , col = x.data.shape
            batch = 1

        else:
            raise ValueError("not supported shape of image_data!")
        

        padded_arr = np.pad(x.data , ((0,0), (self.padding,self.padding),(self.padding  , self.padding) ), mode="constant")
        out_cols = int(np.ceil( (row+2*self.padding - self.kernal_size - 1) /self.stride )+1)
        out_rows = int(np.ceil( (col + 2*self.padding - self.kernal_size - 1)/ self.stride) + 1)

        for i in range(0 , batch):

            for j in range(0 , row):
                if j + self.kernal_size > row + 2*self.padding:
                    break
                for k in range(0 , col):
                    if k + self.kernal_size > col + 2*self.padding:
                        break

                    filter_map = padded_arr[i, j:j+self.kernal_size , k:k+self.kernal_size]
                    val = np.min(filter_map)
                    self.value_list.append(val)
                    K = k+self.stride
                if len(self.value_list ) == batch * out_cols * out_rows :
                    break

                j = j+ self.kernal_size
            if len(self.value_list) == batch * out_cols * out_rows :
                    break
        return Tensor(value=np.array(self.value_list).reshape(batch , out_rows , out_cols))


if __name__ == '__main__':
    print("let us start!")