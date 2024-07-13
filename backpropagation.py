import numpy as np
np.random.seed(1)


def calculate_loss(y, pred):
    # shape of y, pred: (num_of_data, 1)
    return np.square(y - pred).sum() / 2


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        return 1 / (1 + np.exp(-x))


class Linear:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = np.random.normal(0, 0.5, (input_dim, output_dim))
        self.bias = np.random.normal(0, 0.5, output_dim)

        self.weight_grad = None
        self.bias_grad = None

    def forward(self, x):
        # shape of x: (num_of_data, input_dim)
        assert x.shape[1] == self.input_dim
        x = np.dot(x, self.weight) + self.bias.reshape(1, self.output_dim)
        return x # shape: (num_of_data, output_dim)

class NeuralNetwork:
    def __init__(self, struct, n):
        self.struct = struct
        self.n = n
        self.layers = []
        for i in range(1, len(struct)):
            self.layers.append( Linear(struct[i-1], struct[i]) )
            if i != len(struct)-1:
                self.layers.append( Sigmoid() )

    def forward(self, x):
        self.activated_val = [x]
        for layer in self.layers:
            x = layer.forward(x)
            if layer.__class__.__name__ == "Sigmoid":
                self.activated_val.append(x)
        return x

    def backward(self, y, pred, X):

        step = 1
        delta = (pred - y)
        for layer in self.layers[::-1]:
            if layer.__class__.__name__ != "Linear":
                continue
            activated = self.activated_val[-step]
            layer.bias_grad = delta.sum(axis=0)  # columnwise sum
            layer.weight_grad = np.dot(activated.T, delta) # when initial layer, activated value is equal to input matrix, e.g., X
            delta = np.dot(delta, layer.weight.T) * (activated) * (1 - activated)
            step += 1

    def step(self, lr):
        for layer in self.layers:
            if layer.__class__.__name__ == "Linear":
                layer.weight -= lr * layer.weight_grad
                layer.bias -= lr * layer.bias_grad

    def zero_grad(self):
        for layer in self.layers:
            if layer.__class__.__name__ == "Linear":
                layer.weight_grad = None
                layer.bias_grad = None

def train(epoch):
    # n = 1000
    # k = 4
    # struct = [4, 3, 2, 1]
    # X = np.random.randn(n, k)
    # y = np.cos(X).sum(axis=1).reshape(n,1)

    struct = [1, 5, 5, 5, 1]

    n=100
    X = np.linspace(-5, 5, n)
    y = np.sin(X)

    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)

    nn = NeuralNetwork(struct=struct, n=n)
    loss_ = []
    for _ in range(epoch):

        # forward pass
        pred = nn.forward(X)

        # calculate loss
        loss = calculate_loss(y, pred)
        if epoch % 1000 == 0:
            print(loss)
        loss_.append(loss)

        # backward
        nn.backward(y, pred, X)

        # gradient descent
        nn.step(0.001)

        # clear gradient
        nn.zero_grad()

if __name__ == "__main__":
    train(50000)
