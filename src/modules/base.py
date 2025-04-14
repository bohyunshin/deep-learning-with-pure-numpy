from abc import abstractmethod


class BaseModule:
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, dx_out):
        pass

    @abstractmethod
    def get_params_grad(self):
        pass
