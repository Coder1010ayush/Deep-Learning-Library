from Tensor.tensor  import Tensor , toNumpy , toTensor ,visualize_graph
from activation.activation import relu , gelu, sigmoid , tanh , leaky_relu , selu
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import os
import sys
import json

class LinearRegression:
    def __init__(self, input_size, output_size):
        # Initialize weights and biases
        self.weights = Tensor(value=np.random.randn(input_size, output_size))
        self.bias = Tensor(value=np.random.randn(output_size)[0])

    def forward(self, inputs):
        # Forward pass to calculate predictions
        return inputs * self.weights  #+ self.bias

    def mse_loss(self, predictions, targets):
        # Mean Squared Error Loss
        diff = predictions - targets
        val =  (diff.square().sum())
        return val

    def backward(self, loss):
        # Backward pass to compute gradients
        loss.backward()

    def update_parameters(self, learning_rate):
        # Update weights and biases using gradients
        self.weights.data -= learning_rate * self.weights.grad
        self.bias.data -= learning_rate * self.bias.grad

# Example usage:
# Create Linear Regression model
model = LinearRegression(input_size=1, output_size=1)

# Generate some random data
X = np.random.randn(5000, 1)
X.dtype = float
y = 3 * X + 2 + np.random.randn(5000, 1) * 0.1  # Linear relationship with some noise
y.dtype = float
# Convert data to custom tensor class
X_tensor = Tensor(value=X)
y_tensor = Tensor(value=y)
print(X_tensor.data.shape)
print(y_tensor.data.shape)
# Training loop
learning_rate = 0.01
for epoch in range(70):
    # Forward pass
    predictions = model.forward(inputs=X_tensor)
    loss = model.mse_loss(predictions, y_tensor)
    
    # Backward pass
    model.backward(loss)
    # Update parameters
    model.update_parameters(learning_rate)
    
    # Print loss every 100 epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.data}')

# After training, you can use the model to make predictions
new_data = np.array([[1.5], [2.5]])
new_data_tensor = Tensor(value=new_data)
predictions = model.forward(new_data_tensor)
print('Predictions:', (predictions))
out = predictions.data
plt.plot(X.tolist(),y.tolist())
plt.show()
