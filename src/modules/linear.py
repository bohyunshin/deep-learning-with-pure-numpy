from typing import Dict

import numpy as np
from numpy.typing import NDArray

from modules.base import BaseModule


class Linear(BaseModule):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = np.random.uniform(-0.1, 0.1, (input_dim, output_dim))
        self.b = np.random.uniform(-0.1, 0.1, output_dim)

        self.dw = None
        self.db = None

    def forward(self, x: NDArray) -> NDArray:
        # when previous layer is convolution layer
        if len(x.shape) >= 3:
            n, _, _ = x.shape
            x = x.reshape(n, -1)
        elif len(x.shape) == 2:
            assert x.shape[1] == self.input_dim
        else:
            raise ValueError("Unsupported 1 dimensional input")
        self.x = x
        x = np.dot(x, self.w) + self.b
        return x  # shape: (num_of_data, output_dim)

    def backward(self, dx_out: NDArray, **kwargs) -> NDArray:
        self.db = dx_out.sum(axis=0)
        self.dw = np.dot(self.x.T, dx_out)
        dx_in = np.dot(dx_out, self.w.T)
        return dx_in

    def get_params_grad(self) -> Dict[str, Dict]:
        params_info = {
            "w": {"current": self.w, "grad": self.dw},
            "b": {"current": self.b, "grad": self.db},
        }
        return params_info
