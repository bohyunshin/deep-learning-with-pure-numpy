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


def one_hot_vector(n_label: int, y_true: NDArray) -> NDArray:
    """
    params
    ------
    n_label: int
        Number of labels

    y_true: np.ndarray
        True labels, e.g., [0, 3, 2, 0, 1], which is not converted to one hot vector
    """
    return np.eye(n_label)[y_true]
