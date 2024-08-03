import numpy as np


def cross_entropy(y_true, y_pred):
    """
    params
    ------
    y_true: np.ndarray (n, n_label)
    y_pred: np.ndarray (n, n_label)

    returns
    -------
    grad: np.ndarray (n,)
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return -(y_true * np.log2(y_pred)).sum(axis=1) # (n,)
