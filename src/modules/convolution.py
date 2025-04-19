from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from src.modules.base import BaseModule
from src.tools.cnn import calculate_output_dim, calculate_padding_1d, convolve


class Convolution(BaseModule):
    def __init__(
        self,
        input_dim: Tuple[int, int],
        kernel_dim: Tuple[int, int],
        padding: str,
        stride: int,
    ):
        super().__init__()
        """
        Convolution layer with 2 dimensional image.
        """

        if padding not in ["same", "valid"]:
            raise ValueError(
                f"Padding must be one of same or valid, got {padding} instead"
            )
        if stride >= 2:
            raise ValueError("Currently, only stride=1 is supported")
        n, self.h_in, self.w_in = input_dim
        self.padding = padding
        self.stride = stride
        k, _ = kernel_dim

        self.kernel = np.random.uniform(-0.1, 0.1, kernel_dim)
        self.b = np.random.uniform(-0.1, 0.1, 1)

        (h_pad_left, h_pad_right), (w_pad_left, w_pad_right) = self.calculate_pad_dims()
        self.h_pad_left = h_pad_left
        self.h_pad_right = h_pad_right
        self.w_pad_left = w_pad_left
        self.w_pad_right = w_pad_right

        self.h_out = calculate_output_dim(
            input_size=self.h_in,
            pad_size=self.h_pad_left + self.h_pad_right,
            kernel_size=k,
            stride=self.stride,
        )
        self.w_out = calculate_output_dim(
            input_size=self.w_in,
            pad_size=self.w_pad_left + self.w_pad_right,
            kernel_size=k,
            stride=self.stride,
        )

        self.dk = None
        self.db = None

    def forward(self, imgs: NDArray) -> NDArray:
        """
        params
        ------
        imgs: np.ndarray
            dimension (n, h_in, w_in). 3d array input.
        """
        n, h_in, w_in = imgs.shape
        k, _ = self.kernel.shape

        out = np.zeros((n, self.h_out, self.w_out))
        padded_imgs = np.pad(
            imgs,
            pad_width=(
                (0, 0),
                (self.h_pad_left, self.h_pad_right),
                (self.w_pad_left, self.w_pad_right),
            ),
        )
        self.X = padded_imgs
        for i, padded_img in enumerate(padded_imgs):
            out[i] = convolve(
                img=padded_img,
                out_dim=(self.h_out, self.w_out),
                kernel=self.kernel,
                stride=self.stride,
                bias=self.b,
            )
        return out

    def backward(self, dx_out: NDArray) -> NDArray:
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
            dk += convolve(
                img=img,
                out_dim=self.kernel.shape,
                kernel=dX_out_i,
                stride=self.stride,
            )

        rotate_kernel = np.rot90(self.kernel, k=2)
        for i in range(n):
            dx_in[i] = convolve(
                img=dx_out[i],
                out_dim=self.X.shape[1:],
                kernel=rotate_kernel,
                stride=1,
                full=True,
            )

        self.dk = dk
        self.db = db

        return dx_in

    def calculate_pad_dims(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        if self.padding == "same":
            h_f, w_f = self.kernel.shape

            h_pad_left, h_pad_right = calculate_padding_1d(
                padding_type=self.padding,
                input_size=self.h_in,
                kernel_size=h_f,
                stride=self.stride,
            )
            w_pad_left, w_pad_right = calculate_padding_1d(
                padding_type=self.padding,
                input_size=self.w_in,
                kernel_size=w_f,
                stride=self.stride,
            )
            return (h_pad_left, h_pad_right), (w_pad_left, w_pad_right)
        elif self.padding == "valid":
            return (0, 0), (0, 0)
        else:
            raise ValueError(f"Not supported padding type: {self.padding}")

    def get_params_grad(self) -> Dict[str, Dict]:
        params_info = {
            "kernel": {"current": self.kernel, "grad": self.dk},
            "b": {"current": self.b, "grad": self.db},
        }
        return params_info
