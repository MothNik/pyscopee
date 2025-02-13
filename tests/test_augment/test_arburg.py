"""
This test suite implements all tests for the autoregressive model estimation via the
Burg method implemented in the function :func:`pyscopee.augment.extrapolate.arburg`.

"""

# === Imports ===

from array import array

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

    return a


# === Tests ===


def test_arburg_single_segment_different_input_types_against_matlab() -> None:
    """
    Checks the autoregressive model estimation via the function :func:`pyscopee.augment.extrapolate.arburg`
    for a single segment of data against the results from MATLAB.

    The following MATLAB code was used to generate the reference data:

        ```matlab
        rng(1)

        A = [1 -2.7607 3.8106 -2.6535 0.9238];

        y = filter(1,A,0.2*rand(1024,1));

        arcoeffs = arburg(y,4)
        ```

    which gives ``arcoeffs = [1.0000, -2.8980, 3.8834, -2.6198, 0.8028]``.

    Different input types are used to test the function.

    """  # noqa: E501

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

    # first, the slow but very literal implementation is tested
    arcoeffs_ref_slow = arburg_slow(
        xs=np.array([x]),
        x_lens=np.array([x.size]),  # type: ignore
        order=4,
    )
    assert arcoeffs_ref_slow.size == 5, "AR size mismatch for slow implementation"
    assert np.allclose(
        arcoeffs_ref_slow,
        arcoeffs_ref,
        atol=1.0e-4,
        rtol=0.0,
    ), "Results mismatch for slow implementation"

    x_and_type = [
        (x, "numpy"),
        (x.tolist(), "list"),  # type: ignore
        ([x.tolist()], "nested list"),  # type: ignore
        (tuple(x.tolist()), "tuple"),  # type: ignore
        ((tuple(x.tolist()),), "nested tuple"),  # type: ignore
        (array("d", x.tolist()), "Python Array"),  # type: ignore
        (pd.Series(x), "pandas Series"),
    ]

    # the autoregressive model is estimated for all input types
    for x_to_fit, x_type in x_and_type:
        arcoeffs = arburg(
            xs=x_to_fit,
            order=4,
            tikhonov_lambda=None,
        )

        # the results are compared
        assert arcoeffs.size == 5, f"AR size mismatch for {x_type}"
        assert np.allclose(
            arcoeffs,
            arcoeffs_ref,
            atol=1.0e-4,
            rtol=0.0,
        ), f"Results mismatch for {x_type}"

    return


@pytest.mark.parametrize(
    "size, num_segments, order",
    [
        (1024, 10, 10),  # even size, even segments, even order
        (1024, 11, 10),  # even size, odd segments, even order
        (1024, 10, 11),  # even size, even segments, odd order
        (1024, 11, 11),  # even size, odd segments, odd order
        (1024, 1, 100),  # even size, single segment, high even order
        (1024, 1, 101),  # even size, single segment, high odd order
        (1025, 10, 10),  # odd size, even segments, even order
        (1025, 11, 10),  # odd size, odd segments, even order
        (1025, 10, 11),  # odd size, even segments, odd order
        (1025, 11, 11),  # odd size, odd segments, odd order
        (1025, 1, 100),  # odd size, single segment, high even order
        (1025, 1, 101),  # odd size, single segment, high odd order
        (2048, 10, 10),  # even size, even segments, even order
        (2048, 11, 10),  # even size, odd segments, even order
        (2048, 10, 11),  # even size, even segments, odd order
        (2048, 11, 11),  # even size, odd segments, odd order
        (2048, 1, 100),  # even size, single segment, high even order
        (2048, 1, 101),  # even size, single segment, high odd order
        (2049, 10, 10),  # odd size, even segments, even order
        (2049, 11, 10),  # odd size, odd segments, even order
        (2049, 10, 11),  # odd size, even segments, odd order
        (2049, 11, 11),  # odd size, odd segments, odd order
        (2049, 1, 100),  # odd size, single segment, high even order
        (2049, 1, 101),  # odd size, single segment, high odd order
    ],
)
def test_arburg_multi_segments_uniform_size_against_slow(
    size: int,
    num_segments: int,
    order: int,
) -> None:
    """
    Checks the autoregressive model estimation via the function :func:`pyscopee.augment.extrapolate.arburg`
    for multiple equally sized segments of data against the results from a slow but very
    literal implementation.

    """  # noqa: E501

    # the input data and the expected result are defined
    np.random.seed(42)
    segments = np.random.rand(num_segments, size)

    # the autoregressive model is estimated ...
    arcoeffs = arburg(
        xs=segments,
        order=order,
        tikhonov_lambda=None,
    )
    # ... together with the reference results
    arcoeffs_ref = arburg_slow(
        xs=segments,
        x_lens=np.full(num_segments, size),
        order=order,
    )

    # the results are compared
    assert arcoeffs.size == order + 1
    assert np.allclose(
        arcoeffs,
        arcoeffs_ref,
        atol=1e-13,
        rtol=1e-13,
    )

    return


@pytest.mark.parametrize(
    "order",
    [
        10,  # even order
        11,  # odd order
        100,  # high even order
        101,  # high odd order
    ],
)
def test_arburg_multi_segments_different_input_types_variable_size_against_slow(
    order: int,
) -> None:
    """
    Checks the autoregressive model estimation via the function :func:`pyscopee.augment.extrapolate.arburg`
    for multiple segments of data with variable sizes against the results from a slow
    but very literal implementation.

    """  # noqa: E501

    # the input data and the expected result are defined
    np.random.seed(42)
    segments = [
        np.random.rand(128),
        np.random.rand(151),
        np.random.rand(779),
        np.random.rand(284),
        np.random.rand(1005),
        np.random.rand(557),
        np.random.rand(2555),
    ]

    # all the different input types are defined
    input_segments_and_types = [
        (segments, "nested list"),
        (tuple(segments), "tuple of lists"),
        (tuple(tuple(seg) for seg in segments), "nested tuple"),
        ([array("d", seg) for seg in segments], "list or Python Array"),
        (tuple(array("d", seg) for seg in segments), "tuple of Python Arrays"),
        ([pd.Series(seg) for seg in segments], "list of pandas Series"),
        (tuple(pd.Series(seg) for seg in segments), "tuple of pandas Series"),
    ]

    # the reference results are computed after stacking the segments accordingly
    input_segments_stacked = np.empty(
        shape=(len(segments), max(map(len, segments))),
        dtype=np.float64,
    )
    input_segments_lens = np.empty(shape=(len(segments),), dtype=np.int64)
    for iter_i, seg in enumerate(segments):
        input_segments_stacked[iter_i, 0 : len(seg)] = seg
        input_segments_lens[iter_i] = len(seg)

    arcoeffs_ref = arburg_slow(
        xs=input_segments_stacked,
        x_lens=input_segments_lens,
        order=order,
    )

    # the autoregressive model is estimated for all input types
    for input_segments, input_type in input_segments_and_types:
        # to test the function with different integer types for the order, a nested
        # loop is used
        for inner_order in (order, np.int64(order)):
            arcoeffs = arburg(
                xs=input_segments,
                order=inner_order,
                tikhonov_lambda=None,
            )

            # the results are compared
            assert arcoeffs.size == order + 1, f"AR size mismatch for {input_type}"
            assert np.allclose(
                arcoeffs,
                arcoeffs_ref,
                atol=1e-13,
                rtol=1e-13,
            ), f"Results mismatch for {input_type}"

    return


def test_arburg_fails_on_empty_input() -> None:
    """
    Checks that the function :func:`pyscopee.augment.extrapolate.arburg` raises the
    correct exception when the input data is empty.

    """

    x_input = [
        np.array([]),
        [],
        (),
        array("d", []),
        pd.Series([]),
    ]

    for x in x_input:
        with pytest.raises(
            ValueError,
            match="If provided as an Array-Like, 'xs' has to be non-empty.",
        ):
            arburg(
                xs=x,
                order=1,
                tikhonov_lambda=None,
            )

    return


def test_arburg_fails_on_too_small_segments() -> None:
    """
    Checks that the function :func:`pyscopee.augment.extrapolate.arburg` raises the
    correct exception when the segments are too small, i.e., have a size of 1.

    """

    # --- Single segment ---

    np.random.seed(42)
    x_input = np.random.rand(1)

    with pytest.raises(
        ValueError,
        match=(
            "Expected 'xs-segment 0' to have a size between 2 and None for axis 0, but "
            "got a size of 1."
        ),
    ):
        arburg(
            xs=x_input,
            order=1,
            tikhonov_lambda=None,
        )

    # --- Multiple segments ---

    x_input = [  # type: ignore
        np.random.rand(10),
        np.random.rand(20),
        np.random.rand(1),
    ]

    with pytest.raises(
        ValueError,
        match=(
            "Expected 'xs-segment 2' to have a size between 2 and None for axis 0, but "
            "got a size of 1."
        ),
    ):
        arburg(
            xs=x_input,
            order=1,
            tikhonov_lambda=None,
        )

    return


def test_arburg_fails_for_wrong_order() -> None:
    """
    Checks that the function :func:`pyscopee.augment.extrapolate.arburg` raises the
    correct exception when the order is invalid, i.e., either too low or too high.

    """

    np.random.seed(42)
    x_input = np.random.rand(10)

    # the order 1 is tested first; it is invalid independent of the data size
    with pytest.raises(
        ValueError,
        match="Expected 'order' to be >= 1, but got 0.",
    ):
        arburg(
            xs=x_input,
            order=0,
            tikhonov_lambda=None,
        )

    # an order that his too high for the provided data is tested
    with pytest.raises(
        ValueError,
        match="Expected 'order' to be <= 9, but got 11.",
    ):
        arburg(
            xs=x_input,
            order=11,
            tikhonov_lambda=None,
        )

    # an order that is too high for the smallest segment is tested
    x_input = [  # type: ignore
        np.random.rand(20),
        np.random.rand(30),
        np.random.rand(10),
    ]
    with pytest.raises(
        ValueError,
        match="Expected 'order' to be <= 9, but got 10.",
    ):
        arburg(
            xs=x_input,
            order=10,
            tikhonov_lambda=None,
        )

    return


def test_arburg_fails_for_3d_array_input() -> None:
    """
    Checks that the function :func:`pyscopee.augment.extrapolate.arburg` raises the
    correct exception when the input signal is a 3D array.

    """

    np.random.seed(42)
    x_input = np.random.rand(3, 10, 10)

    with pytest.raises(
        ValueError,
        match=(
            "If provided as an Array-Like, 'xs' has to be 1D or 2D, but it is of "
            "dimension 3."
        ),
    ):
        arburg(
            xs=x_input,
            order=1,
            tikhonov_lambda=None,
        )

    return


def test_arburg_tikhonov_regularisation_silent_clipping_and_none() -> None:
    """
    Checks that the regularisation parameter of the autoregressive model estimation of
    the function :func:`pyscopee.augment.extrapolate.arburg` is

    - correctly clipped to zero if the regularisation parameter is negative
    - correctly set to zero if the regularisation parameter is ``None``

    """

    # --- Single segment ---

    np.random.seed(42)
    x = np.random.rand(1024)

    # it is checked whether the regularisation parameter is correctly set to zero
    arcoeffs_standard = arburg(
        xs=x,
        order=10,
        tikhonov_lambda=0.0,
    )
    for lambda_value in [-1.0, None]:
        arcoeffs_regularised = arburg(
            xs=x,
            order=10,
            tikhonov_lambda=lambda_value,
        )

        assert np.array_equal(arcoeffs_standard, arcoeffs_regularised)

    # --- Multiple segments ---

    segments = [
        np.random.rand(128),
        np.random.rand(151),
        np.random.rand(779),
    ]

    # it is checked whether the regularisation parameter is correctly set to zero
    arcoeffs_standard = arburg(
        xs=segments,
        order=10,
        tikhonov_lambda=0.0,
    )
    for lambda_value in [-1.0, None]:
        arcoeffs_regularised = arburg(
            xs=segments,
            order=10,
            tikhonov_lambda=lambda_value,
        )

        assert np.array_equal(arcoeffs_standard, arcoeffs_regularised)

    return


def test_arburg_tikhonov_regularisation_reduces_norm() -> None:
    """
    Checks that the regularisation of the autoregressive model estimation of the
    function :func:`pyscopee.augment.extrapolate.arburg` works as expected, i.e., that
    the norm of the AR coefficients is successively reduced for successively larger
    regularisation parameters.

    """

    # --- Single segment ---

    np.random.seed(42)
    x = np.random.rand(1024)

    # the regularisation is tested to SUCCESSIVELY reduce the norm of the AR
    # coefficients for SUCCESSIVELY larger lambda values
    arcoeffs_standard = arburg(
        xs=x,
        order=10,
        tikhonov_lambda=None,
    )
    previous_norm = np.linalg.norm(arcoeffs_standard)

    for lambda_val in [0.1, 1.0, 10.0, 100.0, 1_000.0]:
        arcoeffs_regularised = arburg(
            xs=x,
            order=10,
            tikhonov_lambda=lambda_val,
        )

        assert previous_norm > np.linalg.norm(arcoeffs_regularised)
        previous_norm = np.linalg.norm(arcoeffs_regularised)

    #  --- Multiple segments ---

    segments = [
        np.random.rand(128),
        np.random.rand(151),
        np.random.rand(779),
    ]

    # the regularisation is tested to SUCCESSIVELY reduce the norm of the AR
    # coefficients for SUCCESSIVELY larger lambda values
    arcoeffs_standard = arburg(
        xs=segments,
        order=10,
        tikhonov_lambda=None,
    )
    previous_norm = np.linalg.norm(arcoeffs_standard)

    for lambda_val in [0.1, 1.0, 10.0, 100.0, 1_000.0]:
        arcoeffs_regularised = arburg(
            xs=segments,
            order=10,
            tikhonov_lambda=lambda_val,
        )

        assert previous_norm > np.linalg.norm(arcoeffs_regularised)
        previous_norm = np.linalg.norm(arcoeffs_regularised)

    return
