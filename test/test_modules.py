import numpy as np
from src.nn.modules import Convolution


def test_convolution():
    n = 100
    h_in, w_in = (15,15)
    kernel_dim = (3,3)
    padding = "same"

    conv = Convolution(kernel_dim, padding)
    pad = conv.calculate_pad_dims()
    imgs = np.random.normal(0, 0.5, (n, h_in, w_in))

    out = conv.forward(imgs)
    assert out.shape == (n, h_in, w_in)

    out_grad = np.random.normal(0, 0.5, out.shape)
    dk, db, dX_in = conv.backward(out_grad)
    assert dk.shape == kernel_dim
    # assert db.shape == ()
    assert dX_in.shape == (n, h_in+pad[0]*2, w_in+pad[1]*2)