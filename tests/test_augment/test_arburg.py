"""
This test suite implements all tests for the autoregressive model estimation via the
Burg method in the module :mod:`pyscopee.augment.extrapolate._numpy_base`.

"""

# === Imports ===

from array import array
from typing import Literal

import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

from pyscopee.augment.extrapolate import arburg

# === Auxiliary Functions ===


def arburg_slow(
    xs: np.ndarray,
    x_lens: np.ndarray,
    order: int,
) -> np.ndarray:
    """
    This function implements the (segmented) Burg method for autoregressive model
    estimation using a slow but very literal implementation.

    For the input arguments, please refer to the documentation of the function
    :func:`pyscopee.augment.extrapolate._numpy_base.arburg_fast`.

    """

    # a nested function to shape the x-data into a matrix with lagged columns
    def shape_to_lagged_x_matrix(
        x: np.ndarray,
        order: int,
    ) -> np.ndarray:
        x_matrix = np.empty(shape=(x.size - order, order + 1), dtype=np.float64)
        for iter_i in range(0, order + 1):
            x_matrix[::, order - iter_i] = x[iter_i : x.size - order + iter_i]

        return x_matrix

    a = np.array([1.0])
    for iter_ord in range(0, order):
        mat_j = np.flip(np.eye(iter_ord + 2), axis=1)
        r_matrix = np.zeros(shape=(iter_ord + 2, iter_ord + 2))
        for iter_i, num_elements in enumerate(x_lens):
            x = xs[iter_i, 0:num_elements]
            x_matrix = shape_to_lagged_x_matrix(x, iter_ord + 1)
            r_matrix += mat_j @ x_matrix.T @ x_matrix @ mat_j + x_matrix.T @ x_matrix

        k_reflect = -(
            np.append(a, 0.0)
            @ mat_j
            @ r_matrix
            @ np.append(a, 0.0)
            / (np.append(a, 0.0) @ r_matrix @ np.append(a, 0.0))
        )

        a = np.append(a, 0.0) + k_reflect * np.flip(np.append(a, 0.0))
        print(iter_ord, k_reflect, a.tolist())

    return a


# === Tests ===


@pytest.mark.parametrize(
    "x_type",
    ["numpy", "array", "pandas"],
)
def test_arburg_single_segment_against_matlab(
    x_type: Literal["numpy", "array", "pandas"],
) -> None:
    """
    This test checks the autoregressive model estimation via the Burg method for a
    single segment of data against the results from MATLAB.

    The following MATLAB code was used to generate the reference data:

        ```matlab
        rng(1)

        A = [1 -2.7607 3.8106 -2.6535 0.9238];

        y = filter(1,A,0.2*rand(1024,1));

        arcoeffs = arburg(y,4)
        ```

    which gives ``arcoeffs = [1.0000, -2.8980, 3.8834, -2.6198, 0.8028]``.

    """

    # the input data and the expected result are defined
    np.random.seed(1)
    # NOTE: the REFERENCE coefficients are the ones against which the results are
    #       compared while the ORIGINAL coefficients are the ones used to generate
    #       the data
    arcoeffs_ref = np.array([1.0000, -2.8980, 3.8834, -2.6198, 0.8028])
    arcoeffs_original = np.array([1.0, -2.7607, 3.8106, -2.6535, 0.9238])
    noise = 0.2 * np.random.rand(1024)
    x = lfilter(
        b=1.0,
        a=arcoeffs_original,
        x=noise,
    )

    # the data is converted to the desired type
    if x_type == "numpy":
        pass
    elif x_type == "array":
        x = array("d", x.tolist())  # type: ignore
    elif x_type == "pandas":
        x = pd.Series(x)
    else:
        raise AssertionError(f"Unknown data type: {x_type}")

    # the autoregressive model is estimated
    arcoeffs = arburg(
        xs=x,
        order=4,
        tikhonov_lambda=None,
    )

    # the results are compared
    assert np.allclose(arcoeffs, arcoeffs_ref, atol=1.0e-4, rtol=0.0)

    return
