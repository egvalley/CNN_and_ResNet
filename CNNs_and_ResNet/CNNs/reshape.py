import numpy as np
from layer import Layer

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    # Mapping from the last convolution-pooling layer to the fully connected layer
    def forward(self, input):
        return np.reshape(input, self.output_shape)
    # Mapping from the fully connected layer to the convolution-pooling layer
    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)