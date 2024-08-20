import numpy as np
from tools.activations import Softmax


class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, y_true, y_prob_pred):
        """
        params
        ------
        y_true: np.ndarray (n, n_label)
        y_prob_pred: np.ndarray (n, n_label)

        returns
        -------
        forward value: np.ndarray (n,)
        """
        n, _ = y_true.shape
        self.n = n
        return -(y_true * np.log(y_prob_pred)).sum() / self.n  # total sum of (n, n_label) ndarray

    def backward(self, y_true, y_prob_pred):
        """
        params
        ------
        y_true: np.ndarray (n, n_label)
        y_prob_pred: np.ndarray (n, n_label)

        returns
        -------
        grad: np.ndarray (n, n_label)
        """
        return -y_true/(y_prob_pred*self.n)


class CrossEntropyLogitLoss:
    def __init__(self):
        self.softmax = Softmax()

    def forward(self, y_true, logit):
        """
        params
        ------
        y_true: np.ndarray (n, n_label)
        logit: np.ndarray (n, n_label)

        returns
        -------
        forward value: np.ndarray (n,)
        """
        n, _ = y_true.shape
        self.n = n
        y_prob_pred = self.softmax.forward(logit)
        self.y_prob_pred = y_prob_pred
        return -(y_true * np.log(y_prob_pred)).sum() / self.n  # total sum of (n, n_label) ndarray

    def backward(self, y_true, logit):
        """
        params
        ------
        y_true: np.ndarray (n, n_label)
        logit: np.ndarray (n, n_label)

        returns
        -------
        grad: np.ndarray (n, n_label)
        """
        return (self.y_prob_pred - y_true) / self.n
