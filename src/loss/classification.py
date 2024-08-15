import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, y_true, y_pred):
        """
        params
        ------
        y_true: np.ndarray (n, n_label)
        y_pred: np.ndarray (n, n_label)

        returns
        -------
        forward value: np.ndarray (n,)
        """
        return -(y_true * np.log2(y_pred)).sum()  # total sum of (n, n_label) ndarray

    def backward(self, y_true, y_pred):
        """
        params
        ------
        y_true: np.ndarray (n, n_label)
        y_pred: np.ndarray (n, n_label)

        returns
        -------
        grad: np.ndarray (n, n_label)
        """
        return -y_true/y_pred + (1-y_true)/(1-y_pred)
