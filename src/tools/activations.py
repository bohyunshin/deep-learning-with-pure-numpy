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
        h,w = x.shape
        out_h = h-self.k+1
        out_w = w-self.k+1
        res = []
        for i in range(out_h):
            for j in range(out_w):
                res.append( x[i:i+self.k , j:j+self.k].max() )
        return np.array(res).reshape((out_h, out_w))
