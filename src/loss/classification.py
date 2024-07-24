import numpy as np


def cross_entropy(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return -(y_true * np.log2(y_pred)).mean()
