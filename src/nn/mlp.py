import numpy as np

from nn.base import BaseNeuralNet
from tools.activations import Sigmoid

np.random.seed(1)


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

class NeuralNetwork(BaseNeuralNet):
    def __init__(self, struct, n):
        super().__init__()
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
