

def forward(network:list, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def backward(network:list, grad, learning_rate):
    for layer in reversed(network):
        grad = layer.backward(grad, learning_rate)
    return grad

    

def train(network:list, x_train:list, y_train:list, epochs:int, learning_rate:int, batch_size:int, verbose = True):
    network[-3].set_mode('train')
    for e in range(epochs):
        r = learning_rate / (3**(e//12))
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = forward(network, x)

            # error
            error += network[-1].cross_entropy(y)

            # backward
            grad = y
            backward(network, grad, r)

        error /= len(x_train)

        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")
