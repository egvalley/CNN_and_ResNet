import numpy as np
from layer import Layer

class Dropout(Layer):
    def __init__(self, rate=0.5):
        super().__init__()
        self.rate = rate
        self.train_mode = True  # To switch between training and evaluation modes
        self.mask = None

    def set_mode(self, mode='train'):
        """Set the mode for the Dropout layer."""
        self.train_mode = mode == 'train'

    # forward propagation
    def forward(self, input):
        self.input = input
        if self.train_mode:
            self.mask = np.random.binomial(1, 1 - self.rate, size=input.shape) / (1 - self.rate)
            self.output = input * self.mask
        else:
            self.output = input
        return self.output

    # backward propagation
    def backward(self, output_gradient, learning_rate=None):  # Learning rate is not used in dropout layer
        if self.train_mode:
            input_gradient = output_gradient * self.mask
        else:
            input_gradient = output_gradient
        return input_gradient

