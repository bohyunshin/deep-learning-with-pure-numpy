import numpy as np

from loss.base import BaseLoss


class MeanSquaredError(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, y, pred):
        n, _ = y.shape
        return np.square(y - pred).sum() / n

    def backward(self, y, pred):
        n, _ = y.shape
        return (pred - y) / n * 2
