import numpy as np


class MeanSquaredError:
    def __init__(self):
        pass

    def forward(self, y, pred):
        n, _ = y.shape
        return np.square(y - pred).sum() / n

    def backward(self, y, pred):
        n, _ = y.shape
        return (pred - y) / n * 2

