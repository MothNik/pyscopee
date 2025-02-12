"""
Module :mod:`signal.smoothing._finite_differences`

This module provides functions for computing the finite difference matrices required
for the Whittaker-Henderson smoother, more specifically the penalty matrix
``D.T @ Q @ D`` where

- ``D`` is a central finite difference matrix
- ``Q`` is a diagonal matrix with the weights of the data points

"""

# === Imports ===

from typing import Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from pyscopee._utils import jit

# === Functions ===


@jit
def _make_central_finite_difference_csr_specs(
    num_points: int,
    order: Literal[2, 4],
) -> Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """
    Generates the specifications for a central finite difference matrix in CSR format.
    A repeating boundary condition is assumed.

    Parameters
    ----------
    num_points : :obj:`int`
        The number of data points.
    order : {``2``, ``4``}
        The order of the finite difference matrix.

    Returns
    -------
    data : :obj:`numpy.ndarray` of shape (m,) of dtype ``numpy.float64``
        The non-zero entries of the matrix.
    indices : :obj:`numpy.ndarray` of shape (m,) of dtype ``numpy.int64``
        The column indices of the non-zero entries.
    indptr : :obj:`numpy.ndarray` of shape (n + 1,) of dtype ``numpy.int64``
        The index pointers for the rows of the matrix.

    """

    # the respective coefficients are extracted together with the number of additional
    # points in the leading and trailing rows that are not just a repetition of the
    # coefficients
    if order == 2:
        coeffs = np.array(
            [1.0, -2.0, 1.0],
            dtype=np.float64,
        )
        num_additional_points = 4

    else:
        coeffs = np.array(
            [1.0, -4.0, 6.0, -4.0, 1.0],
            dtype=np.float64,
        )
        num_additional_points = 14

    # the number of data points is computed to initialise the CSR arrays
    num_coeffs = coeffs.size
    half_num_coeffs_ceil = -(-coeffs.size // 2)  # NOTE: safe ceil division
    num_normal_rows = num_points - order
    num_nonzero_entries = num_normal_rows * num_coeffs + num_additional_points

    data = np.empty(shape=(num_nonzero_entries,), dtype=np.float64)
    indices = np.empty(shape=(num_nonzero_entries,), dtype=np.int64)
    indptr = np.empty(shape=(num_points + 1,), dtype=np.int64)

    # the first rows are handled separately
    data_index_from = 0
    indptr[0] = 0
    for row_index in range(0, order // 2):
        num_summed_points = half_num_coeffs_ceil - row_index
        num_added_points = num_coeffs - num_summed_points + 1

        data_index_to = data_index_from + num_added_points

        data[data_index_from] = coeffs[0:num_summed_points].sum()
        data[data_index_from + 1 : data_index_to] = coeffs[num_summed_points:num_coeffs]

        for column_index, csr_index in enumerate(range(data_index_from, data_index_to)):
            indices[csr_index] = column_index

        indptr[row_index + 1] = indptr[row_index] + num_added_points

        data_index_from = data_index_to

    for row_index in range(order // 2, num_points - order // 2):
        data_index_to = data_index_from + num_coeffs

        data[data_index_from:data_index_to] = coeffs
        for column_index, csr_index in enumerate(range(data_index_from, data_index_to)):
            indices[csr_index] = column_index + (row_index - order // 2)

        indptr[row_index + 1] = indptr[row_index] + num_coeffs

        data_index_from = data_index_to

    # the last rows are handled separately
    for row_index in range(num_points - order // 2, num_points):
        num_summed_points = half_num_coeffs_ceil - (num_points - row_index) + 1
        num_added_points = num_coeffs - num_summed_points + 1

        data_index_to = data_index_from + num_added_points

        data[data_index_from : data_index_to - 1] = coeffs[
            0 : num_coeffs - num_summed_points
        ]
        data[data_index_to - 1] = coeffs[
            num_coeffs - num_summed_points : num_coeffs
        ].sum()

        for column_index, csr_index in enumerate(range(data_index_from, data_index_to)):
            indices[csr_index] = column_index + (row_index - order // 2)

        indptr[row_index + 1] = indptr[row_index] + num_added_points

        data_index_from = data_index_to

    return data, indices, indptr


num_points = 32_000
order = 2

from time import perf_counter_ns

data, indices, indptr = _make_central_finite_difference_csr_specs(
    num_points=num_points,
    order=order,
)

start = perf_counter_ns()
data, indices, indptr = _make_central_finite_difference_csr_specs(
    num_points=num_points,
    order=order,
)
print(f"Took {(1e-3 * (perf_counter_ns() - start)):.0f} mus")

start = perf_counter_ns()
test = csr_matrix((data, indices, indptr), shape=(num_points, num_points))
print(f"Took {(1e-3 * (perf_counter_ns() - start)):.0f} mus")

start = perf_counter_ns()
test.T @ test
print(f"Took {(1e-3 * (perf_counter_ns() - start)):.0f} mus")

print(test.toarray())
