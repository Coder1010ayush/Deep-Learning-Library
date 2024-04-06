import os
import sys
import pathlib
import numpy as np
from Tensor.tensor import visualize_graph as vz , Tensor , toNumpy , toTensor
from nn.linear_nn import Node , Linear,Sequential


model = Sequential(list_of_layers=[(1,5),(5,7),(7,1)])

print(model)

data = np.random.random(size=(5,1))
target = data * 2 + 3

data = Tensor(value=data)
target = Tensor(value=target)
# out = model(x=data)
# print(out)
# loss =model.mse_loss(predictions=out, targets=target)

# print('loss val is ', loss)
# model.backward(loss=loss)
# loss =model.mse_loss(predictions=out, targets=target)

# print('loss val is ', loss)



epochs = 40
learning_rate = 0.01
for epoch in range(epochs):
    # Forward pass
    predictions = model(data)
    loss = model.mse_loss(predictions, target)
    
    # Backward pass
    grad_output = loss.backward()
    #model.zero_grad()  # Reset gradients for next pass
    
    # Update parameters
    for layer in model.layers:
        layer.update_parameters(learning_rate=learning_rate, epoch=epoch)
    
    
    # Print the loss
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.data}')

