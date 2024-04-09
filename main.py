import os
import sys
import pathlib
import numpy as np
from Tensor.tensor import visualize_graph as vz, Tensor, toNumpy, toTensor
from nn.linear_nn import Node, Linear, Sequential
from activation.activation import relu, tanh, selu, gelu, leaky_relu, sigmoid
from regularization.batchnormalization_2d import BatchNorm1d

# model = Sequential(list_of_layers=[(1,5),(5,7),(7,1)])

# print(model)

# data = np.random.random(size=(100,1))
# target = data * 2 + 3

# data = Tensor(value=data)
# target = Tensor(value=target)
# # out = model(x=data)
# # print(out)
# # loss =model.mse_loss(predictions=out, targets=target)

# # print('loss val is ', loss)
# # model.backward(loss=loss)
# # loss =model.mse_loss(predictions=out, targets=target)

# # print('loss val is ', loss)


# epochs = 100
# learning_rate = 0.01
# for epoch in range(epochs):
#     # Forward pass
#     predictions = model(data)
#     loss = model.mse_loss(predictions, target)
#     # Backward pass
#     out = relu(self=loss)
#     out.backward()
#     # Update parameters
#     for layer in model.layers:
#         layer.update_parameters(learning_rate=learning_rate, epoch=epoch)

#    # grad_output = model.backward(loss=loss)
#     # it is not necessary to use model.zero_grad() , but preserve due to pytorch 
#     model.zero_grad()  # Reset gradients for next pass

#     # Print the loss
#     if epoch % 1 == 0:
#         print(f'Epoch {epoch}, Loss: {out.data}')

# testing batch normalization 
# data = Tensor(value=np.random.random(size=(100,40)) )
# bn = BatchNorm1d(momentum=0.8, leaning_rate=0.9)
# bn.initialize_parameters(number_of_features=40)
# out = bn.forward(x=data)
# print('out shape is ', out.data.shape)
# print('out grad shape is ', out.grad.shape)
# print('data shape is ', data.data.shape)
# for params in bn.parameters():
#     print('shape of params is ', params.data.shape)
#
# print('gamma data is ', bn.gamma.data)
# print('beta data is ', bn.beta.data)
# print('gamma grad is ', bn.gamma.grad)
# print()
# print()
# out.backward()
# bn.updateParameters(epoch=1, decay_rate=0.1)
# print('gamma data is ', bn.gamma.data)
# print('beta data is ', bn.beta.data)
# print('gamma grad is ', bn.gamma.grad)
# print()

tensor_obj = Tensor(value=np.random.random(size=(10, 10)))
out = tensor_obj.mean(axis=1)
print(out)
