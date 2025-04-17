import numpy as np
from numpy.typing import NDArray

from loss.base import BaseLoss


class MeanSquaredError(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, y: NDArray, pred: NDArray) -> NDArray:
        n, _ = y.shape
        return np.square(y - pred).sum() / n

    def backward(self, y: NDArray, pred: NDArray) -> NDArray:
        n, _ = y.shape
        return (pred - y) / n * 2
