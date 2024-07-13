from abc import abstractmethod


class BaseNeuralNet:
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, y, pred, X):
        pass

    @abstractmethod
    def step(self, lr):
        pass

    @abstractmethod
    def zero_grad(self):
        pass