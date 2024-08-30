import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))
from tools.utils import convolve


def test_convolution():
    img = np.array(
        [
            [0,1,2],
            [3,4,5],
            [6,7,8]
        ]
    )
    filter = np.array(
        [
            [0,2],
            [1,0]
        ]
    )

    # no padding
    expected = np.array(
        [
            [5,8],
            [14,17]
        ]
    )
    np.testing.assert_array_equal(expected, convolve(img, filter))

    # full convolution
    expected = np.array(
        [
            [0,0,1,2],
            [0,5,8,5],
            [6,14,17,8],
            [12,14,16,0]
        ]
    )
    np.testing.assert_array_equal(expected, convolve(img, filter, full=True))