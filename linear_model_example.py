"""

    train an ann model using nn module    

"""
import torch
from torch.nn import Linear
from torch import nn
import random
import numpy as np
import os
import sys
from nn.linear import Node,Layer,Dense
from Tensor.matrix import Tensor
from sklearn.datasets import make_blobs as mk_blob
from sklearn.datasets import make_moons as mm
import matplotlib.pyplot as plt
from Optimizer.base import SGD
from sklearn.model_selection import train_test_split
# from micrograd.nn import Layer,MLP,Value
def prepare_dataset():
    # preparing or making a dataset
    x , y = mk_blob(n_samples=200, n_features=2, shuffle=True)
    # print('x is : ',x )
    # print()
    # print('y is ',y)

    plt.figure(figsize=(5,5))
    plt.scatter(x[:,0], x[:,1], c=y, s=20, cmap='jet')
    plt.savefig('data_set_distro.png')
    #plt.show() 
    return x , y

class Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(10, 10),
            nn.Linear(10, 10)
        )

    def forward(self,x):
        out = self.dense(x)
        return out


if __name__ == '__main__':
    # model = Node(n_input=100)
    # print(model)
    # print('length of weigths is : ', (model.parameters()[0]).shape())
    # print('length of bias is : ', model.parameters()[1].shape())
    # data = np.random.random(size=(100,10))
    # out = model(x=data)
    # print(out)
    # print(type(out))

    # model = Layer(n_input=20 , n_out=10)
    # params = model.parameters()
    # data = np.random.random(size=(10,10,5,20))
    # print(params)
    # outcome = model(x=data)
    # print(outcome)
    # print('shape of output is : ', outcome.shape())

    # model = Model()
    # params = model.parameters()
    # data = torch.randn(size=(10,5,10))
    # params = model.parameters()
    # outcome = model(data)
    # print(outcome)
    # print(outcome.shape)

    


    model = Dense(n_input=10 , list_layers=[10,10,10])
    data = np.random.random(size=(10,10,5,20))
    params = model.parameters()
    # print(len(params))
    outcome = model(x=data)
    # print(outcome)






    # how to design a mlp layer what would be the best way to take input or information about each layer
















