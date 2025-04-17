from abc import ABC, abstractmethod
from typing import Dict

from numpy.typing import NDArray


class BaseModule(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x: NDArray) -> NDArray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, dx_out: NDArray) -> NDArray:
        raise NotImplementedError

    @abstractmethod
    def get_params_grad(self) -> Dict[str, Dict]:
        raise NotImplementedError
