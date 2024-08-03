import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from src.tools.activations import MaxPooling


def test_max_pooling():
    n = 100
    h_in, w_in = 15, 15
    k = 3
    h_out, w_out = h_in // k, w_in // k

    imgs = np.random.normal(0, 0.5, (n, h_in, w_in))

    max_pooling = MaxPooling(k=k)
    out = max_pooling.forward(imgs)
    assert out.shape == (n, h_out, w_out)
    out_grad = np.random.normal(0, 0.5, out.shape)
    in_grad = max_pooling.backward(out_grad)
    assert in_grad.shape == (n, h_in, w_in)