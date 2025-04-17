from abc import ABC, abstractmethod

from numpy.typing import NDArray


class BaseNeuralNet(ABC):
    def __init__(self):
        self.gradient_step_layers = []

    @abstractmethod
    def forward(self, x: NDArray) -> NDArray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, dx_out: NDArray) -> NDArray:
        raise NotImplementedError

    def step(self, lr: float) -> None:
        for layer in self.gradient_step_layers:
            params_info = layer.get_params_grad()
            for param, info in params_info.items():
                param_grad_step = info["current"] - lr * info["grad"]
                setattr(layer, param, param_grad_step)
