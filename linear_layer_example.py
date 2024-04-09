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
from activation.activation import MeanSquareError
mse = MeanSquareError()
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
    for p in layer.parameters():
        try:
            print(p.data.shape)
        except Exception as e:
            print(p)
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
        loss = mse(prediction, y_train)
        list_result.append(loss.data)
        layer.backward(loss=loss)
        layer.update_parameters(learning_rate=learning_rate,epoch=epochs)
    #d  Check for early stopping
        if loss.data < layer.best_loss:
            layer.best_loss = loss.data
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
    loss = mse(out , y_test)
    print('validation loss is ', loss)

    # random_x = Tensor(value=np.array([[3.0]]))
    # out = layer(x=random_x)
    # print()
    # print(out)
    # loss = layer.mse_loss(predictions=out, targets=Tensor(value=np.array([[17.0]])))
    # print(loss)

    """"
    x_train shape is  (80, 40, 1)
    x_test shape is  (20, 40, 1)
    y_train shape is  (80, 40, 4)
    y_test shape is  (20, 40, 4)


    Size(4, 1)
    Size()
    Linear(in_feature:1 , out_feature:4)
    (80, 40, 4)
    Epoch 0, Loss: 4263.494002059395
    (80, 40, 4)
    Epoch 1, Loss: 4136.636349243948
    (80, 40, 4)
    Epoch 2, Loss: 4012.338696428501
    (80, 40, 4)
    Epoch 3, Loss: 3890.601043613054
    (80, 40, 4)
    Epoch 4, Loss: 3771.4233907976063
    (80, 40, 4)
    Epoch 5, Loss: 3654.80573798216
    (80, 40, 4)
    Epoch 6, Loss: 3540.7480851667133
    (80, 40, 4)
    Epoch 7, Loss: 3429.2504323512667
    (80, 40, 4)
    Epoch 8, Loss: 3320.3127795358205
    (80, 40, 4)
    Epoch 9, Loss: 3213.935126720374
    (80, 40, 4)
    Epoch 10, Loss: 3110.117473904927
    (80, 40, 4)
    Epoch 11, Loss: 3008.85982108948
    (80, 40, 4)
    Epoch 12, Loss: 2910.1621682740333
    (80, 40, 4)
    Epoch 13, Loss: 2814.024515458587
    (80, 40, 4)
    Epoch 14, Loss: 2720.44686264314
    (80, 40, 4)
    Epoch 15, Loss: 2629.429209827693
    (80, 40, 4)
    Epoch 16, Loss: 2540.971557012246
    (80, 40, 4)
    Epoch 17, Loss: 2455.0739041967995
    (80, 40, 4)
    Epoch 18, Loss: 2371.7362513813528
    (80, 40, 4)
    Epoch 19, Loss: 2290.958598565906
    (80, 40, 4)
    Epoch 20, Loss: 2212.740945750459
    (80, 40, 4)
    Epoch 21, Loss: 2137.0832929350127
    (80, 40, 4)
    Epoch 22, Loss: 2063.9856401195657
    (80, 40, 4)
    Epoch 23, Loss: 1993.4479873041187
    (80, 40, 4)
    Epoch 24, Loss: 1925.470334488672
    (80, 40, 4)
    Epoch 25, Loss: 1860.0526816732256
    (80, 40, 4)
    Epoch 26, Loss: 1797.195028857779
    (80, 40, 4)
    Epoch 27, Loss: 1736.8973760423319
    (80, 40, 4)
    Epoch 28, Loss: 1679.159723226885
    (80, 40, 4)
    Epoch 29, Loss: 1623.9820704114386
    (80, 40, 4)
    Epoch 30, Loss: 1571.3644175959917
    (80, 40, 4)
    Epoch 31, Loss: 1521.3067647805449
    (80, 40, 4)
    Epoch 32, Loss: 1473.809111965098
    (80, 40, 4)
    Epoch 33, Loss: 1428.8714591496514
    (80, 40, 4)
    Epoch 34, Loss: 1386.4938063342047
    (80, 40, 4)
    Epoch 35, Loss: 1346.676153518758
    (80, 40, 4)
    Epoch 36, Loss: 1309.418500703311
    (80, 40, 4)
    Epoch 37, Loss: 1274.7208478878645
    (80, 40, 4)
    Epoch 38, Loss: 1242.5831950724178
    (80, 40, 4)
    Epoch 39, Loss: 1213.005542256971
    (80, 40, 4)
    Epoch 40, Loss: 1185.9878894415244
    (80, 40, 4)
    Epoch 41, Loss: 1161.5302366260776
    (80, 40, 4)
    Epoch 42, Loss: 1139.6325838106309
    (80, 40, 4)
    Epoch 43, Loss: 1120.2949309951841
    (80, 40, 4)
    Epoch 44, Loss: 1103.5172781797376
    (80, 40, 4)
    Epoch 45, Loss: 1089.2996253642907
    (80, 40, 4)
    Epoch 46, Loss: 1077.641972548844
    (80, 40, 4)
    Epoch 47, Loss: 1068.5443197333975
    (80, 40, 4)
    Epoch 48, Loss: 1062.0066669179507
    (80, 40, 4)
    Epoch 49, Loss: 1058.0290141025039
    (80, 40, 4)
    Epoch 50, Loss: 1056.6113612870574
    (80, 40, 4)
    Epoch 51, Loss: 1057.7537084716107
    (80, 40, 4)
    Epoch 52, Loss: 1061.3622252107252
    (80, 40, 4)
    Epoch 53, Loss: 1067.149673571968
    (80, 40, 4)
    Epoch 54, Loss: 1074.6553726657835
    (80, 40, 4)
    Epoch 55, Loss: 1083.281868844304
    (80, 40, 4)
    Epoch 56, Loss: 1092.3424930117196
    (80, 40, 4)
    Epoch 57, Loss: 1101.1160194843042
    (80, 40, 4)
    Epoch 58, Loss: 1108.9040755548394
    (80, 40, 4)
    Epoch 59, Loss: 1115.0867319601111
    validation loss is  tensor(287.22581074054335,grad :0)

    out is  tensor([[1.22125218 0.33200886 0.08958438 0.63143448]],grad :[[0. 0. 0. 0.]])
    loss is  tensor(1080.6829051322788,grad :0)

    """
def second_order_solver():
    # preparing the data set for the layer
        # x = np.random.random(size = (100 , 10 , 3))
        # y = np.random.random(size= (100 , 10 , 2))

        x = np.random.randint(low=0, high=4000,size=(100,40,1))
        x.dtype = float
        y = np.random.random(size= (100,40,4))

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
        layer = Linear(in_feature=1, out_feature=4)
        for params in layer.parameters():
            try:
                print(params.shape())
            except Exception as e:
                print(params)
        layer1 = torch.nn.Linear(in_features=1, out_features=1)
        # output = layer1(torch.tensor(data=x_train,dtype=layer1.weight.dtype))
        # print(output)
        print(layer)
        list_result = []
        x_train = Tensor(value=x_train)
        y_train = Tensor(value=y_train)

        learning_rate = 0.01
        epochs = 60
        for epoch in range(0, epochs):

            prediction  = layer(x=x_train)
            print(prediction.data.shape)
            # if epoch == 0:
            #     print(prediction)
            loss = mse(prediction,y_train)
            list_result.append(loss.data)
            layer.backward(loss=loss)
            layer.update_parameters(learning_rate=learning_rate,epoch=epochs)
        #d  Check for early stopping
            # if loss.data < layer.best_loss:
            #     layer.best_loss = loss.data
            #     wait = 0
            # else:
            #     wait += 1
            #     if wait >= layer.patience:
            #         print(f"Early stopping at epoch {epoch}")
            #         break

            if epoch % 1 == 0:
                print(f'Epoch {epoch}, Loss: {loss.data}')


        # plt.plot(list_result)
        # plt.show()


        # testing the x_test data 
        x_test = Tensor(value=x_test)
        y_test = Tensor(value=y_test)
        out = layer(x=x_test)
        loss = mse(out , y_test)
        print('validation loss is ', loss)

        random_x = Tensor(value=np.array([[3.0]]))
        out = layer(x=random_x)
        print()
        print("out is ", out)
        loss = mse(out, Tensor(value=np.array([[17.0]])))
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
    #first_iteration_solver()

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



    