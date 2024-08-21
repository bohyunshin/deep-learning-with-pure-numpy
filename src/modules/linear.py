import numpy as np


class Linear:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = np.random.uniform(-0.1, 0.1, (input_dim, output_dim))
        self.b = np.random.uniform(-0.1, 0.1, output_dim)

        self.dw = None
        self.db = None

    def forward(self, x):
        # shape of x: (num_of_data, input_dim)
        assert x.shape[1] == self.input_dim
        self.x = x
        x = np.dot(x, self.w) + self.b
        return x # shape: (num_of_data, output_dim)

    def backward(self, dx_out, **kwargs):
        """
        params
        ------
        dX_out: np.ndarray(n,)
            Upstream gradient.
        """
        self.db = dx_out.sum()
        self.dw = np.dot(self.x.T, dx_out)
        dx_in = np.dot(dx_out, self.w.T)
        return dx_in

    def get_params_grad(self):
        params_info = {
            "w": {
                "current": self.w,
                "grad": self.dw
            },
            "b": {
                "current": self.b,
                "grad": self.db
            }
        }
        return params_info