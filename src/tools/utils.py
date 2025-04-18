import numpy as np
from numpy.typing import NDArray


def one_hot_vector(n_label: int, y_true: NDArray) -> NDArray:
    """
    params
    ------
    n_label: int
        Number of labels

    y_true: np.ndarray
        True labels, e.g., [0, 3, 2, 0, 1], which is not converted to one hot vector
    """
    return np.eye(n_label)[y_true]
