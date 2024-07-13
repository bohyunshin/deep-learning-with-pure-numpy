from abc import abstractmethod


class BaseNeuralNet:
    def __init__(self):
        pass

    @abstractmethod
    def backpropagate(self):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def zero_grad(self)
        pass: