import numpy as np
from layer import Layer

class MaxPooling(Layer):
    def __init__(self, input_shape:tuple, pooling_kernel_size:int, stride:int, depth:int):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.pooling_kernel_size = pooling_kernel_size
        self.input_shape = input_shape
        self.output_shape = (depth, (input_height - pooling_kernel_size)//stride + 1, (input_width - pooling_kernel_size)//stride + 1)
        self.max_indices = None
        
    def forward(self, input:np.ndarray):
        self.input = input
        self.output = np.random.randn(*self.output_shape)
        self.max_indices = np.zeros((*self.output_shape,2),dtype=np.int8)
        for d in range(self.output_shape[0]):
            for i in range(self.output_shape[1]):
                for j in range(self.output_shape[2]):
                    window = input[d, i:i+self.pooling_kernel_size, j:j+self.pooling_kernel_size]
                    self.output[d, i // self.pooling_kernel_size, j // self.pooling_kernel_size] = np.max(window)
                    # Store the indices of max values for the backward pass
                    index = np.unravel_index(window.argmax(), window.shape)
                    self.max_indices[d, i // self.pooling_kernel_size, j // self.pooling_kernel_size] = [i + index[0], j + index[1]]
        return self.output


    def backward(self, output_gradient, learning_rate):
        input_gradient = np.zeros(self.input_shape)

        for d in range(self.output_shape[0]):
            for i in range(self.output_shape[1]):
                for j in range(self.output_shape[2]):
                    max_i, max_j = self.max_indices[d, i, j]
                    input_gradient[d, max_i, max_j] += output_gradient[d, i, j]
                    

        return input_gradient