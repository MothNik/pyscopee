"""
Module :mod:`signal.smoothing._finite_differences`

This module provides functions for computing the finite difference matrices required
for the Whittaker-Henderson smoother, more specifically the penalty matrix
``D.T @ Q @ D`` where

- ``D`` is a central finite difference matrix
- ``Q`` is a diagonal matrix with the weights of the data points

It heavily relies on a hybrid format between the sparse CSR format and the LAPACK
banded format, named CBR format (compressed banded row).

As an example, the banded matrix

```python
np.array(
    [
        [a00, a01, a02,   0,   0],
        [a10, a11, a12, a13,   0],
        [  0, a21, a22, a23, a24],
        [  0,   0, a32, a33, a34],
        [  0,   0,   0, a43, a44],
    ]
)
```

that consists of 4 diagonals (1 sub-, 1 main-, and 2 super-diagonals) would be stored
in the CBR format as

```python
data = np.array(                    indices = np.array(
    [                                   [
        [a00, a01, a02,   x],               [0, 3],  # for the slice slice(0, 3)
        [a10, a11, a12, a13],               [0, 4],
        [a21, a22, a23, a24],               [1, 5],
        [a32, a33, a34,   x],               [2, 5],
        [a43, a44,   x,   x],               [3, 5],
    ]                                   ]
)                                   )
```

where the entries filled with ``x`` allocated in memory but not used.

So basically, each row is defined by the non-zero entries and the corresponding column
indices in the dense matrix. Since non-zero entries of a banded matrix are always
consecutive for each row, it is sufficient to store 2 indices rather than the specific
one-by-one index storage applied in the CSR format.

The respective dense matrix can be reconstructed as follows:

```python
dense_matrix = np.zeros(
    shape=(data.shape[0], data.shape[0]),
    dtype=np.float64,
)

for row_index, (row_data, row_indices) in enumerate(zip(data, indices)):
    index_from, index_to = row_indices
    num_elements = index_to - index_from
    dense_matrix[row_index, index_from:index_to] = row_data[0:num_elements]
```

"""

# === Imports ===

from typing import Literal, Tuple

import numpy as np
from numba import prange
from numpy.typing import NDArray

from pyscopee._utils import jit

# === Functions ===


@jit(
    "Tuple((float64[:,::1], int64[:,::1]))(int64, int64, boolean)",
    nopython=True,
    # cache=True,
)
def _make_central_finite_difference_specs(
    num_points: int,
    order: Literal[2, 4],
    transpose: bool,
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Generates the CBR-specifications for a square central finite difference
    matrix ``D`` or its transpose ``D.T``.
    A repeating boundary condition is assumed.

    For further details on the CBR format, please refer to the global documentation of
    this module.

    Parameters
    ----------
    num_points : :obj:`int`
        The number of data points.
    order : {``2``, ``4``}
        The order of the finite difference matrix.
    transpose : :obj:`bool`
        Whether the transposed matrix ``D.T`` (``True``) or the original matrix ``D``
        (``False``) should be generated.

    Returns
    -------
    data : :obj:`numpy.ndarray` of shape (num_points, order + 1) and dtype ``numpy.float64``
        The non-zero entries of the finite difference matrix vertically stacked as
        one row for each row of the corresponding dense matrix.
    indices : :obj:`numpy.ndarray` of shape (num_points, 2) and dtype ``numpy.int64``
        The column indices for the non-zero entries in ``data`` vertically stacked as
        one row for each row of the corresponding dense matrix.
        Its first and second column correspond to the ``start`` and ``stop`` of the
        ``slice(start, stop)``, respectively.

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
    # NOTE: for order 2 the matrix is symmetric so the transpose is the same as the
    #       original matrix
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

    # NOTE: for order 4 there is only 2 entries in the leading and trailing rows
    #       that need to be interchanged respectively (denoted by ``flip_value_1`` and
    #       ``flip_value_2`` here)
    else:
        flip_value_1 = -4.0
        flip_value_2 = -3.0
        if transpose:
            flip_value_1, flip_value_2 = flip_value_2, flip_value_1

        # first leading row with repeating boundary condition
        data[0, 0] = 3.0
        data[0, 1] = flip_value_1
        data[0, 2] = 1.0

        indices[0, 0] = 0
        indices[0, 1] = 3

        # second leading row with repeating boundary condition
        data[1, 0] = flip_value_2
        data[1, 1] = 6.0
        data[1, 2] = -4.0
        data[1, 3] = 1.0

        indices[1, 0] = 0
        indices[1, 1] = 4

        # second to last trailing row with repeating boundary condition
        data[num_points - 2, 0] = 1.0
        data[num_points - 2, 1] = -4.0
        data[num_points - 2, 2] = 6.0
        data[num_points - 2, 3] = flip_value_2

        indices[num_points - 2, 0] = num_points - 4
        indices[num_points - 2, 1] = num_points

        # last trailing row with repeating boundary condition
        data[num_points - 1, 0] = 1.0
        data[num_points - 1, 1] = flip_value_1
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


@jit(
    (
        "Tuple((float64[:,::1], int64[:,::1]))"
        "(float64[:,::1], int64[:,::1], int64, float64[:], boolean)"
    ),
    nopython=True,
    # cache=True,
)
def _dot_cbr_finite_difference_matrix_with_diagonal(
    data: NDArray[np.float64],
    indices: NDArray[np.int64],
    order: Literal[2, 4],
    diagonal: NDArray[np.float64],
    multiply_left: bool,
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Computes the matrix product of the finite difference matrix ``A`` with a diagonal
    matrix ``Q`` as either ``Q @ A`` (left multiplication) or ``A @ Q``
    (right multiplication).
    Here, ``A`` can either be the finite difference matrix ``D`` or its transpose
    ``D.T``.

    ``A`` is stored in the CBR format (compressed banded row) and ``Q`` is stored as a
    1D vector with its main diagonal entries.

    For further details on the CBR format, please refer to the global documentation of
    this module.

    Parameters
    ----------
    data : :obj:`numpy.ndarray` of shape (num_points, order + 1) and dtype ``numpy.float64``
        The non-zero entries of the finite difference matrix ``A`` vertically stacked as
        one row for each row of the corresponding dense matrix.
    indices : :obj:`numpy.ndarray` of shape (num_points, 2) and dtype ``numpy.int64``
        The column indices for the non-zero entries in ``data`` vertically stacked as
        one row for each row of the corresponding dense matrix.
        Its first and second column correspond to the ``start`` and ``stop`` of the
        ``slice(start, stop)``, respectively.
    order : {``2``, ``4``}
        The order of the finite difference matrix.
    diagonal : :obj:`numpy.ndarray` of shape (num_points,) and dtype ``numpy.float64``
        The main diagonal of the diagonal matrix ``Q``.
    multiply_left : :obj:`bool`
        Whether the finite difference matrix ``A`` should be multiplied as ``Q @ A``
        (``True``) or ``A @ Q`` (``False``).
        For the CBR format, the left multiplication is way faster than the right
        multiplication.

    Returns
    -------
    new_data : :obj:`numpy.ndarray` of shape (num_points, order + 1) and dtype ``numpy.float64``
        The equivalent to ``data`` for the matrix product ``Q @ A`` or ``A @ Q``.
    indices : :obj:`numpy.ndarray` of shape (num_points, 2) and dtype ``numpy.int64``
        The same as ``indices`` which is not changed when weights are applied.

    """  # noqa: E501

    # the left multiplication allows for an early exit because it is a simple
    # column-wise multiplication
    if multiply_left:
        return (
            data * diagonal[::, np.newaxis],
            indices,
        )

    # the right multiplication is more complicated because different weights are
    # accessed for each row
    weighted_data = np.empty_like(data)

    # the leading ``order // 2`` rows need to be treated separately
    num_points = data.shape[0]
    num_leading_rows = order // 2
    for row_index in range(0, num_leading_rows):
        index_from, index_to = indices[row_index, ::]
        num_elements = index_to - index_from
        weighted_data[row_index, 0:num_elements] = (
            diagonal[0:num_elements] * data[row_index, 0:num_elements]
        )

    # for the central rows, a sliding window stride trick can be applied for very
    # fast computation
    weighted_data[num_leading_rows : num_points - num_leading_rows, ::] = data[
        num_leading_rows : num_points - num_leading_rows, ::
    ] * np.lib.stride_tricks.sliding_window_view(
        diagonal,
        window_shape=(order + 1,),
    )

    # the trailing ``order // 2`` rows need to be treated separately
    for row_index in range(num_points - num_leading_rows, num_points):
        index_from, index_to = indices[row_index, ::]
        num_elements = index_to - index_from
        weighted_data[row_index, 0:num_elements] = (
            diagonal[num_points - num_elements : num_points]
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
def _square_cbr_finite_difference_matrix(
    data: NDArray[np.float64],
    indices: NDArray[np.int64],
    order: Literal[2, 4],
) -> NDArray[np.float64]:
    """
    Computes the squared central finite difference matrix ``A.T @ A`` from the finite
    difference matrix ``A``.
    Here, ``A`` can either be the finite difference matrix ``D`` or its transpose
    ``D.T``.

    ``A`` is stored in the CBR format (compressed banded row).

    For further details on the CBR format, please refer to the global documentation of
    this module.

    Parameters
    ----------
    data : :obj:`numpy.ndarray` of shape (num_points, order + 1) and dtype ``numpy.float64``
        The non-zero entries of the finite difference matrix ``A`` vertically stacked as
        one row for each row of the corresponding dense matrix.
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
        The matrix ``A.T @  A`` in a vertically flipped LAPACK lower symmetric banded
        format.
        Please refer to the Notes section for details.

    Notes
    -----
    For difference order 2, the symmetric squared matrix ``D.T @ D`` that looks like the
    following in its dense form

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

    def convert_to_dense(data, indices, num_points):
        dense_matrix = np.zeros(
            shape=(num_points, num_points),
            dtype=np.float64,
        )

        for row_index, (row_data, row_indices) in enumerate(zip(data, indices)):
            index_from, index_to = row_indices
            num_elements = index_to - index_from
            dense_matrix[row_index, index_from:index_to] = row_data[0:num_elements]

        return dense_matrix

    num_points = 10_000
    order = 4

    data, indices = _make_central_finite_difference_specs(
        num_points=num_points,
        order=order,
        transpose=False,
    )

    start_time = perf_counter_ns()
    data, indices = _make_central_finite_difference_specs(
        num_points=num_points,
        order=order,
        transpose=False,
    )
    stop_time = perf_counter_ns()

    print(f"Took {(1e-3*(stop_time - start_time)):.0f} µs to generate the specs.")

    dense_matrix = convert_to_dense(data, indices, num_points)

    dataT, indicesT = _make_central_finite_difference_specs(
        num_points=num_points,
        order=order,
        transpose=True,
    )

    dense_matrixT = convert_to_dense(dataT, indicesT, num_points)

    assert np.allclose(dense_matrix.T, dense_matrixT)

    np.random.seed(42)
    weights = np.random.rand(num_points)

    weighted_data, indices = _dot_cbr_finite_difference_matrix_with_diagonal(
        data=data,
        indices=indices,
        order=order,
        diagonal=weights,
        multiply_left=False,
    )

    start_time = perf_counter_ns()
    weighted_data, indices = _dot_cbr_finite_difference_matrix_with_diagonal(
        data=data,
        indices=indices,
        order=order,
        diagonal=weights,
        multiply_left=False,
    )
    stop_time = perf_counter_ns()

    print(f"Took {(1e-3*(stop_time - start_time)):.0f} µs to apply the weights.")

    dense_matrix_weighted = convert_to_dense(weighted_data, indices, num_points)

    assert np.allclose(dense_matrix_weighted, dense_matrix * weights[np.newaxis, :])
    print("passed")

    weighted_data, indices = _dot_cbr_finite_difference_matrix_with_diagonal(
        data=data,
        indices=indices,
        order=order,
        diagonal=weights,
        multiply_left=True,
    )

    dense_matrix_weighted = convert_to_dense(weighted_data, indices, num_points)

    assert np.allclose(dense_matrix_weighted, dense_matrix * weights[:, np.newaxis])
    print("passed")

    test = _square_cbr_finite_difference_matrix(
        data=data,
        indices=indices,
        order=order,
    )

    start_time = perf_counter_ns()
    test = _square_cbr_finite_difference_matrix(
        data=data,
        indices=indices,
        order=order,
    )
    stop_time = perf_counter_ns()

    print(
        f"Took {(1e-3*(stop_time - start_time)):.0f} µs to compute the squared matrix."
    )

    print(test)
