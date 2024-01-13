import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activations import Sigmoid,ReLu
from pooling import MaxPooling
from dropout import Dropout
from softmax import Softmax
from network import train, forward

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    two_index = np.where(y == 2)[0][:limit]
    all_indices = np.hstack((zero_index, one_index,two_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float64") / 255
    y = to_categorical(y)  # one-hot code
    y = y.reshape(len(y), 3, 1)
    y = y.astype("float64")
    return x, y

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 200)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# neural network
network = [
    # Input: 28x28 grayscale images, 1 channel
    Convolutional(input_shape=(1, 28, 28), kernel_size=3, depth=16),
    ReLu(),
    MaxPooling(input_shape=(16, 26, 26), pooling_kernel_size=2 , stride=2 ,depth=16), #(26-2)/2 +1 =13

    Convolutional(input_shape=(16, 13, 13), kernel_size=3, depth=32),
    ReLu(),
    MaxPooling(input_shape=(32, 11, 11), pooling_kernel_size=2 ,stride=2 ,depth=32),  # Output: (11-2)/2 + 1 =5

    Reshape(input_shape=(32, 5, 5), output_shape=(32* 5 * 5, 1)),

    Dense(input_size=32 * 5 * 5, output_size=200),
    ReLu(),

    Dropout(0.5),  # Dropout layer to prevent overfitting

    Dense(input_size=200, output_size=3),  # There are 10 digits, so 10 output neurons
    Softmax()  # Softmax activation for multi-class classification
]


# train
train(
    network,
    x_train,
    y_train,
    epochs=60,
    learning_rate=0.0009,
    batch_size=2
)

# test
network[-3].set_mode('test')
for x, y in zip(x_test, y_test):
    output = forward(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")