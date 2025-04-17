from abc import ABC, abstractmethod

from numpy.typing import NDArray


class BaseLoss(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        raise NotImplementedError
