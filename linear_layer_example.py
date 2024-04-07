"""
    this is a demo of how to build a linear layer 
    using the custom nn and tensor class.
    Training and testing and finding the accuracy (r2 score ) of the model.

"""


# importing all the modules that will be needed
import torch
import matplotlib.pyplot as plt
import os
import seaborn
import sys
from Tensor.tensor import Tensor
from Tensor.tensor import visualize_graph, toNumpy,toTensor
from nn.linear_nn import Linear
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np

def first_iteration_solver():
    # preparing the data set for the layer
    x = np.random.randint(low=0, high=2000,size=(2000,1))
    x.dtype = float
    y = x * 3 + 2
    # print(x.shape)
    # print(y.shape)

    # visualising the data set created above using matplotlib.pyplot
    # plt.scatter(x[:,0], x[:,1],c=y, s=20, cmap='jet')
    # plt.show()

    # let us split data into x_train and x_test and y_test , y_train
    x_train , x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,shuffle=True)
    print('x_train shape is ',x_train.shape)
    print('x_test shape is ', x_test.shape)
    print('y_train shape is ', y_train.shape)
    print('y_test shape is ', y_test.shape)
    print()
    print()


    # building a layer
    layer = Linear(in_feature=1, out_feature=1)
    layer1 = torch.nn.Linear(in_features=1, out_features=1)
    # output = layer1(torch.tensor(data=x_train,dtype=layer1.weight.dtype))
    # print(output)
    print(layer)
    list_result = []
    x_train = Tensor(value=x_train)
    y_train = Tensor(value=y_train)

    learning_rate = 0.01
    epochs = 300
    for epoch in range(0, epochs):

        prediction  = layer(x=x_train)
        # if epoch == 0:
        #     print(prediction)
        loss = layer.mse_loss(predictions=prediction, targets=y_train)
        list_result.append(loss.data)
        layer.backward(loss=loss)
        layer.update_parameters(learning_rate=learning_rate,epoch=epochs)
    #d  Check for early stopping
        if loss.data < layer.best_loss:
            best_loss = loss.data
            wait = 0
        else:
            wait += 1
            if wait >= layer.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 1 == 0:
            print(f'Epoch {epoch}, Loss: {loss.data}')


    # plt.plot(list_result)
    # plt.show()


    # testing the x_test data 
    x_test = Tensor(value=x_test)
    y_test = Tensor(value=y_test)
    out = layer(x=x_test)
    loss = layer.mse_loss(out , y_test)
    print('validation loss is ', loss)

    # random_x = Tensor(value=np.array([[3.0]]))
    # out = layer(x=random_x)
    # print()
    # print(out)
    # loss = layer.mse_loss(predictions=out, targets=Tensor(value=np.array([[17.0]])))
    # print(loss)

    """"
    output of the code 
    x_train shape is  (1600, 1)
    x_test shape is  (400, 1)
    y_train shape is  (1600, 1)
    y_test shape is  (400, 1)


    Linear(in_feature:1 , out_feature:1)
    Epoch 0, Loss: 6400.0
    Epoch 1, Loss: 2959.3599999999988
    Epoch 2, Loss: 129.77766399999973
    Epoch 3, Loss: 1243.3691508736015
    Epoch 4, Loss: 4988.769125678455
    Epoch 5, Loss: 6955.3945780843715
    Epoch 6, Loss: 4827.347375338139
    Epoch 7, Loss: 1110.6159033937233
    Epoch 8, Loss: 182.02314361286727
    Epoch 9, Loss: 3135.079929913507
    Epoch 10, Loss: 6492.26659074801
    Epoch 11, Loss: 6300.160114317661
    Epoch 12, Loss: 2784.9850872668394
    Epoch 13, Loss: 86.21162145059617
    Epoch 14, Loss: 1381.9153502141462
    Epoch 15, Loss: 5146.275562565537
    Epoch 16, Loss: 6946.38167243977
    Epoch 17, Loss: 4662.42872484895
    Epoch 18, Loss: 983.9997108760272
    Epoch 19, Loss: 242.81263737551086
    Epoch 20, Loss: 3311.6894021016055
    Epoch 21, Loss: 6576.720726912859
    Epoch 22, Loss: 6193.005723711532
    Epoch 23, Loss: 2612.4071802675135
    Epoch 24, Loss: 51.437941340478766
    Epoch 25, Loss: 1525.895382690902
    Epoch 26, Loss: 5299.458421384524
    Epoch 27, Loss: 6928.379223055731
    Epoch 28, Loss: 4494.440651656509
    Epoch 29, Loss: 863.8487688665832
    Epoch 30, Loss: 311.98857585937384
    Epoch 31, Loss: 3488.73063592017
    Epoch 32, Loss: 6653.143499121371
    Epoch 33, Loss: 6078.814577757259
    Epoch 34, Loss: 2442.073609626201
    Epoch 35, Loss: 25.546758799331258
    Epoch 36, Loss: 1674.9360448103712
    Epoch 37, Loss: 5447.920644452719
    Epoch 38, Loss: 6901.433893187549
    Epoch 39, Loss: 4323.818589304723
    Epoch 40, Loss: 750.4745146566562
    Epoch 41, Loss: 389.37165154891693
    Epoch 42, Loss: 3665.744731577178
    Epoch 43, Loss: 6721.3368157001505
    Epoch 44, Loss: 5957.8826656546225
    Epoch 45, Loss: 2274.4258885342097
    Epoch 46, Loss: 8.605185075908587
    Epoch 47, Loss: 1828.6510156722193
    Epoch 48, Loss: 5591.27741021293
    Epoch 49, Loss: 6865.6155264868885
    validation loss is  tensor(1037.7511996924736,grad :0)

    """
def second_order_solver():
    # preparing the data set for the layer
        # x = np.random.random(size = (100 , 10 , 3))
        # y = np.random.random(size= (100 , 10 , 2))

        x = np.random.randint(low=0, high=4000,size=(100,40,1))
        x.dtype = float
        y = x * 3 + 2

        # print(x.shape)
        # print(y.shape)

        # visualising the data set created above using matplotlib.pyplot
        # plt.scatter(x[:,0], x[:,1],c=y, s=20, cmap='jet')
        # plt.show()

        # let us split data into x_train and x_test and y_test , y_train
        x_train , x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,shuffle=True)
        print('x_train shape is ',x_train.shape)
        print('x_test shape is ', x_test.shape)
        print('y_train shape is ', y_train.shape)
        print('y_test shape is ', y_test.shape)
        print()
        print()


        # building a layer
        layer = Linear(in_feature=1, out_feature=1)
        layer1 = torch.nn.Linear(in_features=1, out_features=1)
        # output = layer1(torch.tensor(data=x_train,dtype=layer1.weight.dtype))
        # print(output)
        print(layer)
        list_result = []
        x_train = Tensor(value=x_train)
        y_train = Tensor(value=y_train)

        learning_rate = 0.01
        epochs = 300
        for epoch in range(0, epochs):

            prediction  = layer(x=x_train)
            print(prediction.data.shape)
            # if epoch == 0:
            #     print(prediction)
            loss = layer.mse_loss(predictions=prediction, targets=y_train)
            list_result.append(loss.data)
            layer.backward(loss=loss)
            layer.update_parameters(learning_rate=learning_rate,epoch=epochs)
        #d  Check for early stopping
            if loss.data < layer.best_loss:
                best_loss = loss.data
                wait = 0
            else:
                wait += 1
                if wait >= layer.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 1 == 0:
                print(f'Epoch {epoch}, Loss: {loss.data}')


        # plt.plot(list_result)
        # plt.show()


        # testing the x_test data 
        x_test = Tensor(value=x_test)
        y_test = Tensor(value=y_test)
        out = layer(x=x_test)
        loss = layer.mse_loss(out , y_test)
        print('validation loss is ', loss)

        random_x = Tensor(value=np.array([[3.0]]))
        out = layer(x=random_x)
        print()
        print("out is ", out)
        loss = layer.mse_loss(predictions=out, targets=Tensor(value=np.array([[17.0]])))
        print("loss is ", loss)

        """"
        output of the code 
        x_train shape is  (1600, 1)
        x_test shape is  (400, 1)
        y_train shape is  (1600, 1)
        y_test shape is  (400, 1)


        """

if __name__ == '__main__':
    second_order_solver()
    first_iteration_solver()

    # arr1 = torch.rand(3,3,requires_grad=True)
    # arr1.retain_grad()
    # arr2 = torch.rand(3,1 , requires_grad=True)
    # arr2.retain_grad()
    # val = arr1 * arr2
    # val.retain_grad()

    # out = val.mean()
    # out.retain_grad()
    # out.backward()


    # print('arr1 grad is ', arr1.grad)
    # print()
    # print('arr2 grad is ', arr2.grad)
    # print()
    # print('val grad is ', val.grad)
    # print()
    # print('out grad is ', out.grad)
    # print()



    