from typing import Tuple

import numpy as np
from numpy.typing import NDArray


class Relu:
    def __init__(self):
        pass

    def forward(self, x: NDArray) -> NDArray:
        self.arg_pos = x > 0
        return x * self.arg_pos

    def backward(self, dx_out: NDArray) -> NDArray:
        arg_pos = self.arg_pos.reshape(dx_out.shape)
        return dx_out * arg_pos


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x: NDArray) -> NDArray:
        self.x = x
        self.frw = 1 / (1 + np.exp(-x))
        return self.frw

    def backward(self, dx_out: NDArray) -> NDArray:
        frw = self.frw.reshape(dx_out.shape)
        return frw * (1 - frw) * dx_out


class MaxPooling:
    def __init__(self, input_dim: Tuple[int, int], k: int):
        self.input_dim = input_dim
        self.k = k
        n, h_in, w_in = input_dim
        h_out, w_out = h_in // k + (h_in % k >= 1), w_in // k + (w_in % k >= 1)
        self.h_out = h_out
        self.w_out = w_out

    def forward(self, x: NDArray) -> NDArray:
        """
        params
        ------
        x: np.ndarray (n, h, w)
            An array of images with 3d dimension
        """
        self._cache = []
        k = self.k
        n, h_in, w_in = x.shape
        h_out, w_out = self.h_out, self.w_out
        out = np.zeros((n, h_out, w_out))
        for i, img in enumerate(x):
            tmp = np.zeros((h_out, w_out))
            cache = []
            for r in range(h_out):
                for c in range(w_out):
                    max_val = img[r * k : (r + 1) * k, c * k : (c + 1) * k].max()
                    row, col = list(zip(*np.where(img == max_val)))[0]
                    tmp[r][c] = max_val
                    cache.append((row, col))
            out[i] = tmp
            self._cache.append(cache)

        return out

    def backward(self, dX_out: NDArray) -> NDArray:
        # when upstream gradient from linear layer is passed
        if len(dX_out.shape) == 2:
            n, _ = dX_out.shape
            dX_out = dX_out.reshape(n, self.h_out, -1)

        n, h_out, w_out = dX_out.shape
        _, h_in, w_in = self.input_dim
        dX_in = np.zeros((n, h_in, w_in))
        for i in range(n):
            c = self._cache[i]
            for j, (row, col) in enumerate(c):
                dX_in[i][row][col] = dX_out[i][j // h_out, j % w_out]
        return dX_in


class Softmax:
    def __init__(self):
        pass

    def forward(self, x: NDArray) -> NDArray:
        x = np.exp(x - x.max(axis=1).reshape(-1, 1))
        y_pred = x / x.sum(axis=1).reshape(-1, 1)
        self.y_pred = y_pred
        return y_pred

    def backward(self, dx_out: NDArray) -> NDArray:
        """
        Step 1
        Calculate jacobian matrix (L, L)
        dyhat_i1/dz_i1 ... dyhat_i1/dz_iL
        ...
        dyhat_iL/dz_i1 ... dyhat_LL/dz_iL

        Note that for each data point, L x L jacobian matrix should be calculated. (N, L, L)

        Step 2
        Calculate downstream gradient
        dL/dz_ik = \sum_j dyhat_ij/dz_ik x dL/dhat_ij

        Note that because it is downstream gradient, its dimension is (N, L)

        params
        ------
        dx_out: np.ndarray (n, n_label)

        returns
        -------
        out_grad: np.ndarray (n, n_label)
        """
        # step 1
        n, L = dx_out.shape
        jacobian = np.zeros((n, L, L))
        identity_mat = np.identity(L)
        for i in range(n):
            y_pred_i = self.y_pred[i]
            right = identity_mat - np.tile(y_pred_i, L).reshape(L, L)
            left = np.tile(y_pred_i, L).reshape(L, L).T
            jacobian[i] = left * right

        # step 2
        dx_in = np.zeros((n, L))
        for i in range(n):
            dL_dz_ij = np.dot(dx_out[i], jacobian[i])  # (1, L)
            dx_in[i] = dL_dz_ij

        return dx_in
