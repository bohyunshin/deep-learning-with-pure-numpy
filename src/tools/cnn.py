from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def convolve(img: NDArray, kernel: NDArray, bias=0, stride=1, full=False) -> NDArray:
    """
    params
    ------
    img: np.ndarray
        dimension (h_in, w_in). one image input assuming no channel
    kernel: np.ndarray
        dimension (h_k, w_k)
    """
    if full and stride != 1:
        raise ValueError(
            f"Stride value of full convolution should be equal to 1, got {stride}"
        )
    k, _ = kernel.shape
    if full:
        img = np.pad(img, pad_width=((k - 1, k - 1), (k - 1, k - 1)))
    # if type(bias) != np.ndarray:
    #     bias = np.zeros_like(kernel)
    h_in, w_in = img.shape
    h_out, w_out = h_in - k + 1, w_in - k + 1
    out = np.zeros((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            out[i, j] = (img[i : i + k, j : j + k] * kernel).sum() + bias
    return out


def calculate_padding_1d(
    padding_type: str, input_size: int, kernel_size: int, stride: int
) -> Tuple[int, int]:
    """
    Calculate padding dimensions for a single dimension based on padding type, input size, kernel size, and stride.

    Args:
        padding_type (str): Either 'same' or 'valid'
        input_size (int): Size of the input dimension
        kernel_size (int): Size of the convolutional kernel
        stride (int): Stride of the convolution

    Returns:
        int or tuple: Padding size (single int) or tuple of (padding_left, padding_right) for asymmetric padding
    """
    if padding_type.lower() == "valid":
        return 0, 0

    elif padding_type.lower() == "same":
        # For 'same' padding, we want the output size to be:
        # output_size = ceil(input_size / stride)

        # Calculate the expected output size
        output_size = (input_size + stride - 1) // stride

        # Calculate the total padding needed
        # derived from standard convolution output formula, e.g.,
        # output_size = (input_size + total_padding - kernel_size) / stride + 1
        padding_needed = max(0, (output_size - 1) * stride + kernel_size - input_size)

        # For even-sized kernels or certain strides, padding might need to be asymmetric
        if padding_needed % 2 == 0:
            # Even padding can be split equally
            padding = padding_needed // 2
            return padding, padding
        else:
            # Odd padding needs to be distributed
            padding_left = padding_needed // 2
            padding_right = padding_needed - padding_left
            return (padding_left, padding_right)
    else:
        raise ValueError("Padding type must be either 'same' or 'valid'")


def calculate_output_dim(
    input_size: int, pad_size: int, kernel_size: int, stride: int
) -> int:
    return (input_size + pad_size - kernel_size) // stride + 1
