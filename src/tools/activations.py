import numpy as np


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        """
        params
        ------
        x: np.ndarray (n,)

        returns
        -------
        out: np.ndarray (n,)
        """
        self.x = x
        return 1 / (1 + np.exp(-x))

    def backward(self):
        """
        returns
        -------
        out: np.ndarray (n,)
            Derivative of sigmoid function.
        """
        frw = self.forward(self.x)
        return frw * (1-frw)


class MaxPooling:
    def __init__(self, input_dim: tuple, k: int):
        self.k = k
        n, h_in, w_in = input_dim
        if h_in % k != 0 or w_in % k != 0:  # no stride assumed
            raise
        h_out, w_out = h_in // k, w_in // k
        self.h_out = h_out
        self.w_out = w_out
        self._cache = []

    def forward(self, x):
        """
        params
        ------
        x: np.ndarray (n, h, w)
            An array of images with 3d dimension
        """
        k = self.k
        n, h_in, w_in = x.shape
        h_out, w_out = self.h_out, self.w_out
        out = np.zeros((n, h_out, w_out))
        for i,img in enumerate(x):
            tmp = np.zeros((h_out, w_out))
            cache = []
            for r in range(h_out):
                for c in range(w_out):
                    max_val = img[r*k:(r+1)*k, c*k:(c+1)*k].max()
                    row,col = list(zip(*np.where(img == max_val)))[0]
                    # row, col = idx // h_in, idx % w_in
                    # max_val = img[row,col]
                    tmp[r][c] = max_val
                    cache.append((row,col))
            out[i] = tmp
            self._cache.append(cache)

        return out

    def backward(self, dX_out):
        n, h_out, w_out = dX_out.shape
        h_in, w_in = h_out * self.k, w_out * self.k
        dX_in = np.zeros((n, h_in, w_in))
        for i in range(n):
            c = self._cache[i]
            for j,(row,col) in enumerate(c):
                dX_in[i][row][col] = dX_out[i][j // h_out, j % w_out]
        return dX_in


class Softmax:
    def __init__(self):
        pass

    def forward(self, x):
        """
        params
        ------
        x: np.ndarray (n, n_label)

        returns
        -------
        y_pred: np.ndarray (n, n_label)
        """
        x = np.exp(x)
        y_pred = x / x.sum(axis=1).reshape(-1,1)
        self.y_pred = y_pred
        return y_pred

    def backward(self, y_true):
        """
        params
        ------
        y_true: np.ndarray (n, n_label)

        returns
        -------
        out_grad: np.ndarray (n, n_label)
        """
        return self.y_pred - y_true