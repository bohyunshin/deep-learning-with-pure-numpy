import numpy as np

from modules.base import BaseModule
from tools.utils import convolve


class Convolution(BaseModule):
    def __init__(self, input_dim, kernel_dim: tuple, padding: str):
        super().__init__()
        n, h_in, w_in = input_dim
        self.padding = padding

        self.kernel = np.random.uniform(-0.1, 0.1, kernel_dim)
        self.b = np.random.uniform(-0.1, 0.1, 1)

        # self.kernel = np.round(self.kernel, 4)
        # self.b = np.round(self.b, 4)

        h_out, w_out = self.calculate_out_dims(h_in, w_in)
        self.h_out = h_out
        self.w_out = w_out

        self.dk = None
        self.db = None

    def forward(self, imgs):
        """
        params
        ------
        imgs: np.ndarray
            dimension (n, h_in, w_in). 3d array input.
        """
        n, h_in, w_in = imgs.shape
        pad = self.calculate_pad_dims()
        out = np.zeros((n, self.h_out, self.w_out))
        imgs = np.pad(imgs, pad_width=((0, 0), (pad[0], pad[0]), (pad[1], pad[1])))
        self.X = imgs
        for i, img in enumerate(imgs):
            out[i] = convolve(img, self.kernel, self.b)
        return out

    def backward(self, dx_out):
        """
        params
        ------
        dx_out: np.ndarray (n, h_out, w_out)
            Upstream gradients from loss function.

        return
        ------
        dX: np.ndarray(n, h_in, w_in)
            Gradients of current input, which is
            full convolution btw 180 degree rotated kernel and upstream gradients
        """

        dx_in = np.zeros_like(self.X)
        n, h_in, w_in = dx_in.shape
        dk = np.zeros_like(self.kernel)
        db = dx_out.sum()

        for img, dX_out_i in zip(self.X, dx_out):
            dk += convolve(img, dX_out_i)

        rotate_kernel = np.rot90(self.kernel, k=2)
        for i in range(n):
            dx_in[i] = convolve(dx_out[i], rotate_kernel, full=True)

        self.dk = dk
        self.db = db

        return dx_in

    def calculate_pad_dims(self):
        if self.padding == "same":
            h_f, w_f = self.kernel.shape
            return (h_f - 1) // 2, (w_f - 1) // 2
        elif self.padding == "valid":
            return 0, 0
        else:
            raise

    def calculate_out_dims(self, h_in, w_in):
        k, _ = self.kernel.shape
        if self.padding == "same":
            return h_in, w_in
        elif self.padding == "valid":
            return h_in - k + 1, w_in - k + 1
        else:
            raise

    def get_params_grad(self):
        params_info = {
            "kernel": {"current": self.kernel, "grad": self.dk},
            "b": {"current": self.b, "grad": self.db},
        }
        return params_info
