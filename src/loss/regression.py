import numpy as np

def mean_squared_error(y, pred):
    # shape of y, pred: (num_of_data, 1)
    return np.square(y - pred).sum() / 2