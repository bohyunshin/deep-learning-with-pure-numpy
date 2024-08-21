from abc import abstractmethod


class BaseLoss:
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, y_true, y_pred):
        pass

    @abstractmethod
    def backward(self, y_true, y_pred):
        pass