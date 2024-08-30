import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))
from modules.convolution import Convolution
from modules.linear import Linear


def test_convolution():
    n = 100
    h_in, w_in = (15,15)
    kernel_dim = (3,3)
    padding = "same"

    conv = Convolution(input_dim=(n, h_in, w_in),
                       kernel_dim=kernel_dim,
                       padding=padding)
    pad = conv.calculate_pad_dims()
    imgs = np.random.normal(0, 0.5, (n, h_in, w_in))

    out = conv.forward(imgs)
    assert out.shape == (n, h_in, w_in)

    out_grad = np.random.normal(0, 0.5, out.shape)
    dX_in = conv.backward(out_grad)
    assert conv.dk.shape == kernel_dim
    # assert db.shape == ()
    assert dX_in.shape == (n, h_in+pad[0]*2, w_in+pad[1]*2)
