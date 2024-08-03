import numpy as np

from tools.utils import convolve


class Linear:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = np.random.normal(0, 0.5, (input_dim, output_dim))
        self.bias = np.random.normal(0, 0.5, output_dim)

        self.weight_grad = None
        self.bias_grad = None

    def forward(self, x):
        # shape of x: (num_of_data, input_dim)
        assert x.shape[1] == self.input_dim
        x = np.dot(x, self.weight) + self.bias.reshape(1, self.output_dim)
        return x # shape: (num_of_data, output_dim)


class Convolution:
    def __init__(self, input_dim, kernel_dim: tuple , padding: str):
        n, h_in, w_in = input_dim
        self.padding = padding
        self.kernel = np.random.normal(0, 0.5, kernel_dim)
        self.bias = np.random.normal(0, 0.5, 1)

        h_out, w_out = self.calculate_out_dims(h_in, w_in)
        self.h_out = h_out
        self.w_out = w_out

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
        imgs = np.pad(imgs, pad_width=((0,0),(pad[0], pad[0]), (pad[1], pad[1])))
        self.X = imgs
        for i, img in enumerate(imgs):
            out[i] = convolve(img, self.kernel, self.bias)
        return out

    def backward(self, dX_out):
        """
        params
        ------
        dX_out: np.ndarray (n, h_out, w_out)
            Upstream gradients of next layers. dimension
        dk: np.ndarray (h_k, w_k)
            Gradients of kernel, which is
            convolution btw kernel and upstream gradients
        db: np.ndarray
            Gradients of bias, which is
            sum of upstream gradients

        return
        ------
        dX: np.ndarray(n, h_in, w_in)
            Gradients of current input, which is
            full convolution btw 180 degree rotated kernel and upstream gradients
        """

        dX_in = np.zeros_like(self.X)
        n, h_in, w_in = dX_in.shape
        dk = np.zeros_like(self.kernel)
        db = dX_out.sum()

        for img, dX_out_i in zip(self.X, dX_out):
            dk += convolve(img, dX_out_i)

        rotate_kernel = np.rot90(self.kernel, k=2)
        for i in range(n):
            dX_in[i] = convolve(dX_out[i], rotate_kernel, full=True)

        return dk, db, dX_in

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
            return h_in-k+1, w_in-k+1
        else:
            raise
