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
        n, self.h_in, self.w_in = input_dim
        self.padding = padding
        self.stride = stride

        self.kernel = np.random.uniform(-0.1, 0.1, kernel_dim)
        self.b = np.random.uniform(-0.1, 0.1, 1)

        # h_out, w_out = self.calculate_out_dims()
        # self.h_out = h_out
        # self.w_out = w_out

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
        k, _ = self.kernel
        (h_pad_left, h_pad_right), (w_pad_left, w_pad_right) = self.calculate_pad_dims()

        h_out = calculate_output_dim(
            input_size=h_in,
            pad_size=h_pad_left + h_pad_right,
            kernel_size=k,
            srtide=self.stride,
        )
        w_out = calculate_output_dim(
            input_size=w_in,
            pad_size=w_pad_left + w_pad_right,
            kernel_size=k,
            srtide=self.stride,
        )

        out = np.zeros((n, h_out, w_out))
        padded_imgs = np.pad(
            imgs,
            pad_width=((0, 0), (h_pad_left, h_pad_right), (w_pad_left, w_pad_right)),
        )
        self.X = padded_imgs
        for i, padded_img in enumerate(padded_imgs):
            out[i] = convolve(padded_img, self.kernel, self.b)
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
            dk += convolve(img, dX_out_i)

        rotate_kernel = np.rot90(self.kernel, k=2)
        for i in range(n):
            dx_in[i] = convolve(dx_out[i], rotate_kernel, full=True)

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

    # def calculate_out_dims(self, h_pad_total: int, w_pad_total: int) -> Tuple[int, int]:
    #     k, _ = self.kernel.shape
    #     if self.padding == "same":
    #         h_out = calculate_output_dim(
    #             input_size=self.h_in,
    #             pad_size=h_pad_total,
    #             kernel_size=k,
    #             srtide=self.stride,
    #         )
    #         w_out =
    #         return h_out, w_out
    #     elif self.padding == "valid":
    #         return h_in - k + 1, w_in - k + 1
    #     else:
    #         raise

    def get_params_grad(self) -> Dict[str, Dict]:
        params_info = {
            "kernel": {"current": self.kernel, "grad": self.dk},
            "b": {"current": self.b, "grad": self.db},
        }
        return params_info


if __name__ == "__main__":
    imgs = np.random.uniform(size=(10, 6, 6))
    k = 3
    stride = 1
    padding = "same"
    conv = Convolution((10, 6, 6), 3, padding, stride)
    conv.forward(imgs)
