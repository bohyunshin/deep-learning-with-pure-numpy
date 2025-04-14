import os
import sys

sys.path.append(os.path.join(os.getcwd(), "src"))
import numpy as np

from tools.activations import MaxPooling, Relu, Softmax


def test_relu():
    imgs = np.array([[-1, 1, -1], [1, -1, 1], [-1, 1, -1]]).reshape((1, 3, 3))

    relu = Relu()
    out = relu.forward(imgs)
    out_expected = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]).reshape((1, 3, 3))

    # forward test
    np.testing.assert_array_equal(out_expected, out)

    # backward test
    out_grad = np.random.normal(0, 0.5, out.shape)
    in_grad = relu.backward(out_grad)

    in_grad_expected = np.zeros_like(imgs, dtype=np.float64)
    in_grad_expected[0][0][1] = in_grad[0][0][1]
    in_grad_expected[0][1][0] = in_grad[0][1][0]
    in_grad_expected[0][1][2] = in_grad[0][1][2]
    in_grad_expected[0][2][1] = in_grad[0][2][1]
    np.testing.assert_array_equal(in_grad_expected, in_grad)


def test_max_pooling_equal_size():
    n = 1
    k = 2

    imgs = np.arange(16).reshape(n, 4, 4)

    max_pooling = MaxPooling(input_dim=(n, 4, 4), k=k)
    out = max_pooling.forward(imgs)
    out_expected = np.array([[5, 7], [13, 15]]).reshape(n, 2, 2)

    # forward test
    np.testing.assert_array_equal(out_expected, out)

    # backward test
    out_grad = np.random.normal(0, 0.5, out.shape)
    in_grad = max_pooling.backward(out_grad)

    in_grad_expected = np.zeros_like(imgs, dtype=np.float64)
    in_grad_expected[0][1][1] = out_grad[0][0][0]
    in_grad_expected[0][1][3] = out_grad[0][0][1]
    in_grad_expected[0][3][1] = out_grad[0][1][0]
    in_grad_expected[0][3][3] = out_grad[0][1][1]
    np.testing.assert_array_equal(in_grad_expected, in_grad)


def test_max_pooling_unequal_size():
    n = 1
    k = 3

    imgs = np.arange(16).reshape(n, 4, 4)

    max_pooling = MaxPooling(input_dim=(n, 4, 4), k=k)
    out = max_pooling.forward(imgs)
    out_expected = np.array([[10, 11], [14, 15]]).reshape(n, 2, 2)

    # forward test
    np.testing.assert_array_equal(out_expected, out)

    # backward test
    out_grad = np.random.normal(0, 0.5, out.shape)
    in_grad = max_pooling.backward(out_grad)

    in_grad_expected = np.zeros_like(imgs, dtype=np.float64)
    in_grad_expected[0][2][2] = out_grad[0][0][0]
    in_grad_expected[0][2][3] = out_grad[0][0][1]
    in_grad_expected[0][3][2] = out_grad[0][1][0]
    in_grad_expected[0][3][3] = out_grad[0][1][1]
    np.testing.assert_array_equal(in_grad_expected, in_grad)


def test_softmax():
    n = 4
    L = 3

    logits = np.random.normal(0, 0.5, (n, L))
    dx_out = np.random.normal(0, 0.5, (n, L))

    softmax = Softmax()

    # forward
    out = softmax.forward(logits)

    # backward
    dx_in = softmax.backward(dx_out)

    identity_mat = np.identity(L)
    for i in range(n):
        yhat = out[i]
        mat = np.array([yhat, yhat, yhat])
        right = identity_mat - mat
        left = mat.T
        jacobian = left * right
        grad = np.dot(dx_out[i], jacobian)
        np.testing.assert_array_almost_equal(dx_in[i], grad)
