import numpy as np

from nn.base import BaseNeuralNet
from tools.activations import Sigmoid
from nn.modules import Linear

np.random.seed(1)


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
