from abc import abstractmethod


class BaseNeuralNet:
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, dx_out):
        pass

    @abstractmethod
    def step(self, lr):
        pass
