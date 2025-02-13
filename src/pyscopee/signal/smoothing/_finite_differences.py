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
from numba import prange
from numpy.typing import NDArray

from pyscopee._utils import jit

# === Functions ===


@jit(
    "Tuple((float64[:,::1], int64[:,::1]))(int64, int64)",
    nopython=True,
    # cache=True,
)
def _make_transposed_central_finite_difference_specs(
    num_points: int,
    order: Literal[2, 4],
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Generates the specifications for a transposed square central finite difference
    matrix ``D.T`` in a hybrid format between the sparse CSR format and the LAPACK
    banded format.
    A repeating boundary condition is assumed.

    Parameters
    ----------
    num_points : :obj:`int`
        The number of data points.
    order : {``2``, ``4``}
        The order of the finite difference matrix.

    Returns
    -------
    data : :obj:`numpy.ndarray` of shape (num_points, order + 1) and dtype ``numpy.float64``
        The non-zero entries of the finite difference matrix vertically stacked as
        one row for each row of the corresponding dense matrix.
        Please refer to the Notes section for details.
    indices : :obj:`numpy.ndarray` of shape (num_points, 2) and dtype ``numpy.int64``
        The column indices for the non-zero entries in ``data`` vertically stacked as
        one row for each row of the corresponding dense matrix.
        Its first and second column correspond to the ``start`` and ``stop`` of the
        ``slice(start, stop)``, respectively.
        Please refer to the Notes section for details.

    Notes
    -----
    Given that it fits into memory, the transposed dense matrix can be reconstructed as
    follows:

    ```python
    transposed_dense_matrix = np.zeros(
        shape=(num_points, num_points),
        dtype=np.float64,
    )

    for row_index, (row_data, row_indices) in enumerate(zip(data, indices)):
        dense_index_from, dense_index_to = row_indices
        num_elements = index_to - index_from
        transposed_dense_matrix[row_index, dense_index_from:dense_index_to] = (
            row_data[0:num_elements]
        )
    ```

    So, the transposed dense matrix

    ```python
    np.array(
        [
            [-1, 1, 0, 0, 0],
            [1, -2, 1, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 1, -2, 1],
            [0, 0, 0, 1, -1],
        ]
    )
    ```

    would be stored as

    ```python
    data = np.array(
        [
            [-1, 1, x],
            [1, -2, 1],
            [1, -2, 1],
            [1, -2, 1],
            [1, -1, x],
        ]
    )

    indices = np.array(
        [
            [0, 2],
            [0, 3],
            [1, 4],
            [2, 5],
            [3, 5],
        ]
    )
    ```

    where the entries filled with ``x`` are not used.

    """  # noqa: E501

    # the respective coefficients for the normal as well as the leading and trailing
    # rows are extracted based on the difference order
    data = np.empty(
        shape=(num_points, order + 1),
        dtype=np.float64,
    )
    indices = np.empty(
        shape=(num_points, 2),
        dtype=np.int64,
    )

    # the coefficients are extracted and the leading and trailing rows are already
    # pre-filled based on the difference order
    if order == 2:
        # leading row with repeating boundary condition
        data[0, 0] = -1.0
        data[0, 1] = 1.0

        indices[0, 0] = 0
        indices[0, 1] = 2

        # trailing row with repeating boundary condition
        data[num_points - 1, 0] = 1.0
        data[num_points - 1, 1] = -1.0

        indices[num_points - 1, 0] = num_points - 2
        indices[num_points - 1, 1] = num_points

        # finally, the central coefficients are obtained
        central_coefficients = np.array([1.0, -2.0, 1.0], dtype=np.float64)

    else:
        # first leading row with repeating boundary condition
        data[0, 0] = 3.0
        data[0, 1] = -3.0
        data[0, 2] = 1.0

        indices[0, 0] = 0
        indices[0, 1] = 3

        # second leading row with repeating boundary condition
        data[1, 0] = -4.0
        data[1, 1] = 6.0
        data[1, 2] = -4.0
        data[1, 3] = 1.0

        indices[1, 0] = 0
        indices[1, 1] = 4

        # second to last trailing row with repeating boundary condition
        data[num_points - 2, 0] = 1.0
        data[num_points - 2, 1] = -4.0
        data[num_points - 2, 2] = 6.0
        data[num_points - 2, 3] = -4.0

        indices[num_points - 2, 0] = num_points - 4
        indices[num_points - 2, 1] = num_points

        # last trailing row with repeating boundary condition
        data[num_points - 1, 0] = 1.0
        data[num_points - 1, 1] = -3.0
        data[num_points - 1, 2] = 3.0

        indices[num_points - 1, 0] = num_points - 3
        indices[num_points - 1, 1] = num_points

        # finally, the central coefficients are obtained
        central_coefficients = np.array([1.0, -4.0, 6.0, -4.0, 1.0], dtype=np.float64)

    # the central coefficients are filled in
    num_leading_rows = order // 2
    data[num_leading_rows : num_points - num_leading_rows, :] = central_coefficients
    indices[num_leading_rows : num_points - num_leading_rows, 0] = np.arange(
        0,
        num_points - order,
        dtype=np.int64,
    )
    indices[num_leading_rows : num_points - num_leading_rows, 1] = np.arange(
        order + 1,
        num_points + 1,
        dtype=np.int64,
    )

    return data, indices


# @jit(
#     "Tuple((float64[:,::1], int64[:,::1]))(float64[:,::1], int64[:,::1], int64, float64[:])",
#     nopython=True,
#     # cache=True,
# )
def _right_apply_weights_central_finite_difference_matrix(
    data: NDArray[np.float64],
    indices: NDArray[np.int64],
    order: Literal[2, 4],
    weights: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Applies the weights to the non-zero entries of the transposed central finite
    difference matrix ``D.T`` to compute the matrix ``D.T @ Q`` where ``Q`` is a
    diagonal matrix with the weights of the data points.

    For the matrix specifications, please refer to the documentation of
    :func:`_make_transposed_central_finite_difference_specs`.

    Parameters
    ----------
    data : :obj:`numpy.ndarray` of shape (num_points, order + 1) and dtype ``numpy.float64``
        The non-zero entries of the transposed finite difference matrix vertically
        stacked as one row for each row of the corresponding dense matrix.
    indices : :obj:`numpy.ndarray` of shape (num_points, 2) and dtype ``numpy.int64``
        The column indices for the non-zero entries in ``data`` vertically stacked as
        one row for each row of the corresponding dense matrix.
        Its first and second column correspond to the ``start`` and ``stop`` of the
        ``slice(start, stop)``, respectively.
    order : {``2``, ``4``}
        The order of the finite difference matrix.
    weights : :obj:`numpy.ndarray` of shape (num_points,) and dtype ``numpy.float64``
        The weights of the data points.

    Returns
    -------
    weighted_data : :obj:`numpy.ndarray` of shape (num_points, order + 1) and dtype ``numpy.float64``
        The equivalent to ``data`` with the weights applied.
    indices : :obj:`numpy.ndarray` of shape (num_points, 2) and dtype ``numpy.int64``
        The same as ``indices`` which is not changed when weights are applied.

    """  # noqa: E501

    weighted_data = np.empty_like(data)

    # the leading ``order // 2`` rows need to be treated separately
    num_points = data.shape[0]
    num_leading_rows = order // 2
    for row_index in range(0, num_leading_rows):
        index_from, index_to = indices[row_index, ::]
        num_elements = index_to - index_from
        weighted_data[row_index, 0:num_elements] = (
            weights[0:num_elements] * data[row_index, 0:num_elements]
        )

    # for the central rows, a sliding window stride trick can be applied for very
    # fast computation
    weighted_data[num_leading_rows : num_points - num_leading_rows, ::] = data[
        num_leading_rows : num_points - num_leading_rows, ::
    ] * np.lib.stride_tricks.sliding_window_view(
        weights,
        window_shape=(order + 1,),
    )

    # the trailing ``order // 2`` rows need to be treated separately
    for row_index in range(num_points - num_leading_rows, num_points):
        index_from, index_to = indices[row_index, ::]
        num_elements = index_to - index_from
        weighted_data[row_index, 0:num_elements] = (
            weights[num_points - num_elements : num_points]
            * data[row_index, 0:num_elements]
        )

    return weighted_data, indices


@jit(
    "Tuple((int64, int64, int64, int64))(int64, int64, int64, int64)",
    nopython=True,
    inline="always",
    # cache=True,
)
def _get_dot_overlap_indices(
    row_index_from: int,
    row_index_to: int,
    column_index_from: int,
    column_index_to: int,
) -> Tuple[int, int, int, int]:

    dot_index_from = max(row_index_from, column_index_from)
    dot_index_to = min(row_index_to, column_index_to)

    return (
        dot_index_from - row_index_from,
        dot_index_to - row_index_from,
        dot_index_from - column_index_from,
        dot_index_to - column_index_from,
    )


@jit(
    "float64[:,::1](float64[:,::1], int64[:,::1], int64)",
    nopython=True,
    # parallel=True,
    # cache=True,
)
def _square_transposed_central_finite_difference_matrix(
    data: NDArray[np.float64],
    indices: NDArray[np.int64],
    order: Literal[2, 4],
) -> NDArray[np.float64]:
    """
    Computes the squared central finite difference matrix ``D.T @ D`` from the
    specifications of the transposed central finite difference matrix ``D.T``.

    For the matrix specifications, please refer to the documentation of
    :func:`_make_transposed_central_finite_difference_specs`.

    Parameters
    ----------
    data : :obj:`numpy.ndarray` of shape (num_points, order + 1) and dtype ``numpy.float64``
        The non-zero entries of the transposed finite difference matrix vertically
        stacked as one row for each row of the corresponding dense matrix.
    indices : :obj:`numpy.ndarray` of shape (num_points, 2) and dtype ``numpy.int64``
        The column indices for the non-zero entries in ``data`` vertically stacked as
        one row for each row of the corresponding dense matrix.
        Its first and second column correspond to the ``start`` and ``stop`` of the
        ``slice(start, stop)``, respectively.
    order : {``2``, ``4``}
        The order of the finite difference matrix.

    Returns
    -------
    squared_data : :obj:`numpy.ndarray` of shape (num_points, order + 1) and dtype ``numpy.float64``
        The matrix ``D.T @  D`` in a vertically flipped LAPACK lower symmetric banded
        format.
        Please refer to the Notes section for details.

    Notes
    -----
    For difference order 2, the symmetric squared matrix that looks like the following
    in its dense form

    ```python
    np.array(
        [
            [a00, a01, a02,   0,   0],
            [a01, a11, a12, a13,   0],
            [a02, a12, a22, a23, a24],
            [  0, a13, a23, a33, a34],
            [  0,   0, a24, a34, a44],
        ]
    )
    ```

    would be stored in the LAPACK lower symmetric banded format as

    ```python
    np.array(
        [
            [a00, a11, a22, a33, a44],
            [a01, a12, a23, a34,   x],
            [a02, a13, a24,   x,   x],
        ]
    )
    ```

    where the entries filled with ``x`` are not used.
    This function however returns

    ```python
    np.array(
        [
            [a00, a01, a02],
            [a11, a12, a13],
            [a22, a23, a24],
            [a33, a34,   x],
            [a44,   x,   x],
        ]
    )
    ```

    so basically, the same matrix as the LAPACK matrix with a C-style row-major
    ordering rather than a Fortran-style column-major ordering.

    """  # noqa: E501

    # the number of diagonals (including the main diagonal) in the upper part is given
    # by the order; this value determined how many columns need to be considered for
    # each row (except for the last ``order`` rows)
    bandwidth = order + 1

    # the matrix product is computed row by row for the upper triangular part only
    # due to symmetry
    squared_data = np.empty_like(data)
    num_points = data.shape[0]
    for row_index in prange(0, num_points):
        dense_index_from, dense_index_to = indices[row_index, ::]
        num_elements = dense_index_to - dense_index_from

        # the product of the row with itself is computed directly
        squared_data[row_index, 0] = np.dot(
            data[row_index, 0:num_elements],
            data[row_index, 0:num_elements],
        )

        # the product with the following columns is computed
        for col_index in range(
            row_index + 1,
            min(row_index + bandwidth, num_points),
        ):
            (
                row_dot_index_from,
                row_dot_index_to,
                column_dot_index_from,
                column_dot_index_to,
            ) = _get_dot_overlap_indices(
                row_index_from=dense_index_from,
                row_index_to=dense_index_to,
                column_index_from=indices[col_index, 0],
                column_index_to=indices[col_index, 1],
            )

            squared_data[row_index, col_index - row_index] = np.dot(
                data[row_index, row_dot_index_from:row_dot_index_to],
                data[col_index, column_dot_index_from:column_dot_index_to],
            )

    return squared_data


if __name__ == "__main__":

    from time import perf_counter_ns

    num_points = 10_000
    order = 2

    data, indices = _make_transposed_central_finite_difference_specs(
        num_points=num_points,
        order=order,
    )

    start_time = perf_counter_ns()
    data, indices = _make_transposed_central_finite_difference_specs(
        num_points=num_points,
        order=order,
    )
    stop_time = perf_counter_ns()

    print(f"Took {(1e-3*(stop_time - start_time)):.0f} µs to generate the specs.")

    dense_matrix = np.zeros(
        shape=(num_points, num_points),
        dtype=np.float64,
    )

    for row_index, (row_data, row_indices) in enumerate(zip(data, indices)):
        index_from, index_to = row_indices
        num_elements = index_to - index_from
        dense_matrix[row_index, index_from:index_to] = row_data[0:num_elements]

    print("both close?", np.allclose(dense_matrix.T, dense_matrix))

    np.random.seed(42)
    weights = np.random.rand(num_points)

    weighted_data, indices = _right_apply_weights_central_finite_difference_matrix(
        data=data,
        indices=indices,
        order=order,
        weights=weights,
    )

    start_time = perf_counter_ns()
    weighted_data, indices = _right_apply_weights_central_finite_difference_matrix(
        data=data,
        indices=indices,
        order=order,
        weights=weights,
    )
    stop_time = perf_counter_ns()

    print(f"Took {(1e-3*(stop_time - start_time)):.0f} µs to apply the weights.")

    dense_matrix_weighted = np.zeros(
        shape=(num_points, num_points),
        dtype=np.float64,
    )

    for row_index, (row_data, row_indices) in enumerate(zip(weighted_data, indices)):
        index_from, index_to = row_indices
        num_elements = index_to - index_from
        dense_matrix_weighted[row_index, index_from:index_to] = row_data[0:num_elements]

    assert np.allclose(dense_matrix_weighted, dense_matrix * weights[np.newaxis, :])
    print("passed")

    test = _square_transposed_central_finite_difference_matrix(
        data=data,
        indices=indices,
        order=order,
    )

    start_time = perf_counter_ns()
    test = _square_transposed_central_finite_difference_matrix(
        data=data,
        indices=indices,
        order=order,
    )
    stop_time = perf_counter_ns()

    print(
        f"Took {(1e-3*(stop_time - start_time)):.0f} µs to compute the squared matrix."
    )

    print(test)
