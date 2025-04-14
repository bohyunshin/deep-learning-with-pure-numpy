import numpy as np

from modules.base import BaseModule


class Linear(BaseModule):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = np.random.uniform(-0.1, 0.1, (input_dim, output_dim))
        self.b = np.random.uniform(-0.1, 0.1, output_dim)

        # self.w = np.round(self.w, 4)
        # self.b = np.round(self.b, 4)

        self.dw = None
        self.db = None

    def forward(self, x):
        # shape of x: (num_of_data, input_dim)
        assert x.shape[1] == self.input_dim
        self.x = x
        x = np.dot(x, self.w) + self.b
        return x  # shape: (num_of_data, output_dim)

    def backward(self, dx_out, **kwargs):
        """
        params
        ------
        dx_out: np.ndarray(n,)
            Upstream gradient from loss function.
        """
        # self.db = dx_out.sum()
        self.db = dx_out.sum(axis=0)
        self.dw = np.dot(self.x.T, dx_out)
        dx_in = np.dot(dx_out, self.w.T)
        return dx_in

    def get_params_grad(self):
        params_info = {
            "w": {"current": self.w, "grad": self.dw},
            "b": {"current": self.b, "grad": self.db},
        }
        return params_info
