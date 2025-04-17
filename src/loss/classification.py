import numpy as np
from numpy.typing import NDArray

from loss.base import BaseLoss
from tools.activations import Softmax


class CrossEntropyLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, y_true: NDArray, y_prob_pred: NDArray) -> NDArray:
        n, _ = y_true.shape
        self.n = n
        return (
            -(y_true * np.log(y_prob_pred)).sum() / self.n
        )  # total sum of (n, n_label) ndarray

    def backward(self, y_true: NDArray, y_prob_pred: NDArray) -> NDArray:
        return -y_true / (y_prob_pred * self.n)


class CrossEntropyLogitLoss(BaseLoss):
    def __init__(self):
        self.softmax = Softmax()

    def forward(self, y_true: NDArray, logit: NDArray) -> NDArray:
        n, _ = y_true.shape
        self.n = n
        y_prob_pred = self.softmax.forward(logit)
        self.y_prob_pred = y_prob_pred
        return (
            -(y_true * np.log(y_prob_pred)).sum() / self.n
        )  # total sum of (n, n_label) ndarray

    def backward(self, y_true: NDArray, logit: NDArray) -> NDArray:
        return (self.y_prob_pred - y_true) / self.n
