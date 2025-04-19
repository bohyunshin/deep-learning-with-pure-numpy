import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from modules.convolution import Convolution


def test_convolution():
    n = 100
    h_in, w_in = (15, 15)
    kernel_dim = (3, 3)
    padding = "same"
    stride = 1

    conv = Convolution(
        input_dim=(n, h_in, w_in), kernel_dim=kernel_dim, padding=padding, stride=stride
    )
    (h_pad_left, h_pad_right), (w_pad_left, w_pad_right) = conv.calculate_pad_dims()
    imgs = np.random.normal(0, 0.5, (n, h_in, w_in))

    out = conv.forward(imgs)
    assert out.shape == (n, h_in, w_in)

    out_grad = np.random.normal(0, 0.5, out.shape)
    dX_in = conv.backward(out_grad)

    # check gradient dimension
    assert conv.dk.shape == kernel_dim
    assert dX_in.shape == (
        n,
        h_in + h_pad_left + h_pad_right,
        w_in + w_pad_left + w_pad_right,
    )
