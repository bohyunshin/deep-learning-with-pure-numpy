import numpy as np


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        return 1 / (1 + np.exp(-x))


class MaxPooling:
    def __init__(self, k: int):
        self.k = k
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
        if h_in % k != 0 or w_in % k != 0:
            raise
        h_out, w_out = h_in // k, w_in // k
        out = []
        for img in x:
            tmp = np.zeros((h_out, w_out))
            cache = []
            for i in range(h_out):
                for j in range(w_out):
                    idx = img[i*k:(i+1)*k, j*k:(j+1)*k].argmax()
                    row, col = idx // h_in, idx % w_in
                    max_val = img[row,col]
                    tmp[i][j] = max_val
                    cache.append((row,col))
            out.append(tmp)
            self._cache.append(cache)

        return np.ndarray(out)

    def backward(self, dX_out):
        n, h_out, w_out = dX_out.shape
        h_in, w_in = h_out * self.k, w_out * self.k
        dX_in = np.zeros((n, h_in, w_in))
        for i in range(n):
            c = self._cache[i]
            for j,(row,col) in enumerate(c):
                dX_in[i][row][col] = dX_out[i][j // self.k, j % self.k]
        return dX_in


class Softmax:
    def __init__(self):
        pass

    def forward(self, x):
        x = x.reshape(-1,)
        x = np.exp(x - x.max())
        denom = x.sum()
        return x / denom