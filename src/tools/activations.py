import numpy as np


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        return 1 / (1 + np.exp(-x))


class MaxPooling:
    def __init__(self, k: int):
        self.k = k

    def forward(self, x):
        h, w = x.shape
        k = self.k
        if h % k != 0:
            raise
        return x.reshape(h // k, k, w // k, k).max(axis=(1, 3))


class Softmax:
    def __init__(self):
        pass

    def forward(self, x):
        x = x.reshape(-1,)
        x = np.exp(x - x.max())
        denom = x.sum()
        return x / denom