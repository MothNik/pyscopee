"""
Module :mod:`augment.extrapolate._extrapolate`

This module contains the actual functions to extrapolate signals beyond their original
range that include input validation and implementation selection.

Currently, the following extrapolation methods are available:

- Burg's method for autoregressive model estimation
- Ordinary Least Squares (OLS) for autoregressive model estimation

"""

# === Setup ===

__all__ = [
    "arburg",
    "ar_ordinary_least_squares",
    "extrapolate_autoregressive",
]

# === Imports ===

from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..._utils import (
    Integer,
    RealNumeric,
    get_validated_integer,
    get_validated_real_numeric,
    get_validated_real_numeric_1d_array_like,
    warn_verbose,
)
from ._autoregressive_base import (
    ar_one_step_least_squares as _ar_one_step_least_squares,
)
from ._autoregressive_base import arburg_fast as _arburg_fast
from ._autoregressive_base import (
    extrapolate_autoregressive as _extrapolate_autoregressive,
)

# === Auxiliary Functions ===


def _prepare_x_segments_for_ar_fit(
    xs: Union[ArrayLike, List[ArrayLike], Tuple[ArrayLike, ...]]
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Prepares the input signal segments for the autoregressive model estimation by

    - validating them
    - stacking them row-wise into a 2D-Array,
    - determining the lengths of the segments

    Parameters
    ----------
    xs : Array-like of shape (n,) or (m, n) or list or tuple of (n_i,)-Array-likes
        The real input signal (segments) for which the AR coefficients are to be
        computed.
        For details, see the docstring of, e.g., :func:`arburg`.

    Returns
    -------
    xs_packaged : :class:`numpy.ndarray` of shape (len(xs), max(len(xs[i]))) of dtype ``numpy.float64``
        The segments stacked row-wise into a 2D-Array. Its ``i``-th row corresponds to
        ``xs[i]`` with the remaining elements padded to ``max(len(xs[i]))`` with
        arbitrary values (``numpy.empty`` initialisation). Please refer to the
        Notes section for more details.
    x_lens : :class:`numpy.ndarray` of shape (len(xs),) of dtype :class:`numpy.int64`
        The lengths of the segments. Its ``i``-th element corresponds to ``len(xs[i])``.
        Please refer to the Notes section for more details.

    Raises
    ------
    TypeError
        If ``xs`` is not of the expected type.
    ValueError
        If ``xs`` is an empty Array-like.
    ValueError
        If ``xs`` is not a real numeric 1D Array-like or iterable of real numeric 1D
        Array-likes with the expected size.

    Notes
    -----
    For extracting the ``i``-th segment from ``xs_packaged``, the following code can be
    used:

    ```python
    segment_i = xs_packaged[i, 0:x_lens[i]]
    ```

    """  # noqa: E501

    # the conversion to an iterable for the actual validation depends on the type of
    # ``xs``
    # first, an attempt is made to convert the input to an at-least-1D NumPy Array
    try:
        xs = np.atleast_1d(np.asarray(xs, dtype=np.float64))

    # NOTE: this can happen for an iterable of Array-likes of inconsistent size
    except ValueError:
        pass

    # now, the more specific type-based validation is performed
    # Case 1: xs is now a NumPy-1D-Array
    if isinstance(xs, np.ndarray):
        # empty Arrays are invalid
        if xs.size < 1:
            raise ValueError("If provided as an Array-Like, 'xs' has to be non-empty.")

        if xs.ndim not in (1, 2):
            raise ValueError(
                f"If provided as an Array-Like, 'xs' has to be 1D or 2D, but it is of "
                f"dimension {xs.ndim}."
            )

        if xs.ndim == 1:
            xs = (xs,)

    # Case 2: xs is not a list or tuple of Array-likes of inconsistent size, i.e., it
    #         is not a supported type
    elif not isinstance(xs, (list, tuple)):  # pragma: no cover
        x_type_name = f"{type(xs)}"[7:-1]  # removes the "<class '" and "'>" parts
        raise ValueError(
            f"Expected 'xs' to be an Array-like or a list or tuple of Array-likes, "
            f"but got an object of type {x_type_name}."
        )

    xs = [
        get_validated_real_numeric_1d_array_like(
            value=x,
            name=f"xs-segment {index}",
            min_size=2,
            max_size=None,
            output_dtype=np.float64,
        )
        for index, x in enumerate(xs)  # type: ignore
    ]

    x_lens = np.array([x.size for x in xs], dtype=np.int64)  # type: ignore

    # the segments are stacked row-wise into a (partially empty) 2D-Array
    xs_packaged = np.empty(
        shape=(len(xs), np.max(x_lens)),
        dtype=np.float64,
    )
    for index, (segment, segment_len) in enumerate(zip(xs, x_lens)):
        xs_packaged[index, 0:segment_len] = segment

    return xs_packaged, x_lens


def _prepare_ar_coeffs_for_extrapolation(
    ar_coeffs: Union[ArrayLike, Tuple[ArrayLike, ArrayLike], List[ArrayLike]],
    signal_size: int,
    zero_lag_warn: bool = True,
) -> Tuple[NDArray[np.float64], int, int]:
    """
    Prepares the AR coefficients for the extrapolation by

    - validating them
    - normalising them if necessary
    - determining the order(s) of the AR model(s)
    - stacking them row-wise into a 2D-Array

    Parameters
    ----------
    ar_coeffs : Array-like of shape (order + 1,) or 2-tuple or 2-list of Array-likes with shapes (order1 + 1,) and (order2 + 1,)
        The AR coefficients of the autoregressive model.
        For details, see the docstring of, e.g., :func:`extrapolate_autoregressive`.
    signal_size : :class:`int`
        The size of the signal to be extrapolated.
    zero_lag_warn : :class:`bool`, default=``True``
        Whether to issue a warning if the zero-lag coefficient of the AR model is not
        exactly equal to ``1.0`` (``True``) or not (``False``).
        Setting this to ``False`` will only disable the warning and not the
        normalisation.

    Returns
    -------
    ar_coeffs_internal : :class:`numpy.ndarray` of shape (2, max(order1, order2) + 1) of dtype ``numpy.float64``
        The AR coefficients of the autoregressive models stacked row-wise into a
        2D-Array. The first row corresponds to the left hand side model and the second
        row to the right hand side model. The remaining elements are padded to
        ``max(order1, order2) + 1`` with arbitrary values (``numpy.empty``
        initialisation). Its ``i``-th column corresponds to the ``i``-th lag.
        Please refer to the Notes section for more details.
        If the zero-lag coefficient is not exactly 1.0, all coefficients are normalised
        by this value and a warning is issued (see ``zero_lag_warn``).
    ar_order_left, ar_order_right : :class:`int`
        The orders of the autoregressive models for the left and right hand side
        extrapolation, respectively. Please refer to the Notes section for more details.

    Raises
    ------
    TypeError
        If ``ar_coeffs`` is not of the expected type.
    ValueError
        If ``ar_coeffs`` is not of expected size.

    Notes
    -----
    For extracting the AR coefficients of the left hand side model from
    ``ar_coeffs_internal``, the following code can be used:

    ```python
    ar_coeffs_left = ar_coeffs_internal[0, 0:ar_order_left + 1]
    ar_coeffs_right = ar_coeffs_internal[1, 0:ar_order_right + 1]
    ```

    """  # noqa: E501

    # for a single AR model, the same coefficients are used for both sides
    if not isinstance(ar_coeffs, (list, tuple)):
        ar_coeffs = (ar_coeffs, ar_coeffs)

    if len(ar_coeffs) != 2:
        raise ValueError(
            f"Expected 'ar_coeffs' to be an Array-like or a 2-tuple or 2-list of "
            f"Array-likes, but got an object of length {len(ar_coeffs)}."
        )

    ar_coeffs_left, ar_coeffs_right = [
        get_validated_real_numeric_1d_array_like(
            value=coeffs,
            name=f"ar_coeffs[{index}]",
            min_size=2,
            max_size=signal_size + 1,
            output_dtype=np.float64,
        )
        for index, coeffs in enumerate(ar_coeffs)
    ]

    ar_order_left = ar_coeffs_left.size - 1
    ar_order_right = ar_coeffs_right.size - 1
    ar_coeffs_internal = np.empty(
        shape=(2, max(ar_order_left, ar_order_right) + 1),
        dtype=np.float64,
    )

    ar_coeffs_internal[0, 0 : ar_order_left + 1] = ar_coeffs_left
    ar_coeffs_internal[1, 0 : ar_order_right + 1] = ar_coeffs_right

    # if the zero-lag coefficient is not exactly 1.0, a scaling is performed and a
    # warning is issued if requested
    for coeffs in ar_coeffs_internal:
        if coeffs[0] != 1.0:
            warn_verbose(
                f"The zero-lag coefficient of the AR model is not exactly 1.0, but "
                f"{coeffs[0]:.5e}.\n"
                f"All coefficients are normalised by this value.\n"
                f"This warning can be suppressed by setting 'zero_lag_warn=False'.",
                RuntimeWarning,
                issue_warning=zero_lag_warn,
            )

            coeffs /= coeffs[0]

    return ar_coeffs_internal, ar_order_left, ar_order_right


def _get_validated_pad_width(
    pad_width: Union[Integer, Tuple[Integer, Integer], List[Integer]],
) -> Tuple[int, int]:
    """
    Validates the padding width for extrapolation.

    Parameters
    ----------
    pad_width : :class:`int` or 2-tuple or 2-list of :class:`int`
        The size of the extrapolation on the left and right side.
        For details, see the documentation of, e.g., :func:`extrapolate_autoregressive`.

    Returns
    -------
    pad_width_left : :class:`int`
        The size of the extrapolation on the left side.
    pad_width_right : :class:`int`
        The size of the extrapolation on the right side.

    Raises
    ------
    TypeError
        If ``pad_width`` is not of the expected type.
    ValueError
        If ``pad_width`` is not within the expected range.

    """  # noqa: E501

    if not isinstance(pad_width, (list, tuple)):
        pad_width = (pad_width, pad_width)

    if len(pad_width) != 2:
        raise ValueError(
            f"Expected 'pad_width' to be an integer or a 2-tuple or 2-list of "
            f"integers, but got an object of length {len(pad_width)}."
        )

    return tuple(  # type: ignore
        get_validated_integer(
            value=value,
            name=f"pad_width[{index}]",
            min_value=0,
            max_value=None,
            clip=True,
        )
        for index, value in enumerate(pad_width)
    )


# === Functions ===


def arburg(
    xs: Union[ArrayLike, List[ArrayLike], Tuple[ArrayLike, ...]],
    order: Integer = 1,
    tikhonov_lambda: Optional[RealNumeric] = None,
) -> NDArray[np.float64]:
    """
    Computes the AR coefficients for an autoregressive model using a fast implementation
    of Burg's method that relies on an implicit matrix formulation that even allows for
    Tikhonov regularisation.

    If available at runtime, a Numba-accelerated implementation is used instead of the
    NumPy-based one.

    Parameters
    ----------
    xs : Array-like of shape (n,) or (m, n) or list or tuple of (n_i,)-Array-likes
        The real input signal (segments) for which the AR coefficients are to be
        computed.
        2D-ArrayLikes are interpreted as row-wise stacked segments.
        If multiple segments are provided, they are treated as individual segments of
        a single signal and the resulting AR model will minimise the forward and
        backward prediction errors over all segments combined. However, this does not
        mean that the AR model is fitted to the concatenated signal, i.e., no forward
        or backward prediction is performed across the segments.
        Its/their data type is internally promoted to ``numpy.float64``.
        Each of them must hold at least ``2`` elements.
    order : :class:`int`, default=``1``
        The order of the autoregressive model.
        It has to be within the range ``[1, min(len(xs[i]) - 1)]`` for all ``xs[i]``.
    tikhonov_lambda : :class:`float` or :class:`int` or ``None``, default=``None``
        The Tikhonov regularisation parameter lambda. It has to be non-negative
        (``lam >= 0.0``) and if ``> 0.0``, it will result in Tikhonov regularisation.
        Values ``< 0.0`` are silently clipped to ``0.0``.
        Higher values of lambda lead to a more stable solution but may introduce a bias.
        ``None`` is equivalent to ``0.0``.

    Returns
    -------
    a_prediction : :class:`numpy.ndarray` of shape (order  + 1,) of dtype ``numpy.float64``
        The AR coefficients of the autoregressive model.
        To be consistent with Matlab's ``arburg`` function, the zero-lag coefficient is
        included in the output as the first element ``a_prediction[0]`` which is always
        ``1.0``.
        Its ``i``-th element corresponds to the coefficient of the ``i``-th lag
        starting from ``0`` for the zero-lag coefficient.

    Raises
    ------
    TypeError
        If ``xs``, ``order``, or ``tikhonov_lambda`` are not of the expected type.
    ValueError
        If ``xs`` is an empty Array-like.
    ValueError
        If ``xs``is not a real numeric 1D Array-like or iterable of real numeric 1D
        Array-likes with the expected size.
    ValueError
        If ``order`` is not within the allowed range.

    References
    ----------
    The implementation is based on the pseudo-code provided in [1]_ and extended to
    a segmented version using the idea described in [2]_.

    .. [1] Vos K., A Fast Implementation of Burg's Method (2013)
    .. [2] De Waele S., Broersen P.M.T, The Burg Algorithm for Segments, IEEE
       Transactions on Signal Processing (2000), 48(10), pp. 2876-2880,
       DOI: 10.1109/78.869039

    """  # noqa: E501

    # --- Input Validation ---

    xs, x_lens = _prepare_x_segments_for_ar_fit(xs=xs)

    order = get_validated_integer(
        value=order,
        name="order",
        min_value=1,
        max_value=int(np.min(x_lens) - 1),
    )

    tikhonov_lambda = get_validated_real_numeric(
        value=tikhonov_lambda if tikhonov_lambda is not None else 0.0,
        name="tikhonov_lambda",
    )

    # --- Computation ---

    # depending on the choice of the user, the Numba-accelerated or the NumPy-based
    # implementation is used
    return _arburg_fast(
        xs=xs,
        x_lens=x_lens,
        order=order,
        tikhonov_lambda=tikhonov_lambda,
    )


def ar_ordinary_least_squares(
    xs: Union[ArrayLike, List[ArrayLike], Tuple[ArrayLike, ...]],
    order: Integer = 1,
    tikhonov_lambda: Optional[RealNumeric] = None,
    lstsq_solver: Literal["symmetric", "sym", "positive_definite", "pos"] = "symmetric",
) -> NDArray[np.float64]:
    """
    Computes the AR coefficients for an autoregressive model using an Ordinary Least
    Squares (OLS) approach with optional Tikhonov regularisation.

    If available at runtime, a Numba-accelerated implementation is used instead of the
    NumPy-based one.

    Parameters
    ----------
    xs : Array-like of shape (n,) or (m, n) or list or tuple of (n_i,)-Array-likes
        The real input signal (segments) for which the AR coefficients are to be
        computed.
        2D-ArrayLikes are interpreted as row-wise stacked segments.
        If multiple segments are provided, they are treated as individual segments of
        a single signal and the resulting AR model will minimise the forward and
        backward prediction errors over all segments combined. However, this does not
        mean that the AR model is fitted to the concatenated signal, i.e., no forward
        or backward prediction is performed across the segments.
        Its/their data type is internally promoted to ``numpy.float64``.
        Each of them must hold at least ``2`` elements.
    order : :class:`int`, default=``1``
        The order of the autoregressive model.
        It has to be within the range ``[1, min(len(xs[i]) - 1)]`` for all ``xs[i]``.
        rcond : :class:`float`
    tikhonov_lambda : :class:`float` or :class:`int` or ``None``, default=``None``
        The Tikhonov regularisation parameter lambda. It has to be non-negative
        (``lam >= 0.0``) and if ``> 0.0``, it will result in Tikhonov regularisation.
        Values ``< 0.0`` are silently clipped to ``0.0``.
        Higher values of lambda lead to a more stable solution but may introduce a bias.
        ``None`` is equivalent to ``0.0``.
        A value of ``0.0`` corresponds to the standard OLS approach, but this may lead
        to numerical instability.
    lstsq_solver : {``"sym"``, ``"symmetric"``, ``"pos"``, ``"positive_definite"``}, default=``"symmetric"``
        The solver to use for the least squares problem, which can be

        - ``"sym"`` or ``"symmetric"``: Symmetric indefinite factorisation which is a
            slower but more stable solver.
        - ``"pos"`` or ``"positive_definite"``: Cholesky factorisation which is the
            a very fast but less stable solver.

    Returns
    -------
    a_prediction : :class:`numpy.ndarray` of shape (order  + 1,) of dtype ``numpy.float64``
        The AR coefficients of the autoregressive model.
        To be consistent with Matlab's ``arburg`` function, the zero-lag coefficient is
        included in the output as the first element ``a_prediction[0]`` which is always
        ``1.0``.
        Its ``i``-th element corresponds to the coefficient of the ``i``-th lag
        starting from ``0`` for the zero-lag coefficient.

    Raises
    ------
    TypeError
        If ``xs``, ``order``, or ``tikhonov_lambda`` are not of the expected type.
    ValueError
        If ``xs`` is an empty Array-like.
    ValueError
        If ``xs``is not a real numeric 1D Array-like or iterable of real numeric 1D
        Array-likes with the expected size.
    ValueError
        If ``order`` is not within the allowed range.
    ValueError
        If ``lstsq_solver`` is not one of the supported solvers.
    numpy.linalg.LinAlgError
        If the matrix inversion fails because ``tikhonov_lambda`` is too low to make
        the design matrix non-singular.

    """  # noqa: E501

    # --- Input Validation ---

    xs, x_lens = _prepare_x_segments_for_ar_fit(xs=xs)

    order = get_validated_integer(
        value=order,
        name="order",
        min_value=1,
        max_value=int(np.min(x_lens) - 1),
    )

    # NOTE: the factor of 2 is required because the AR model is fitted in the forward
    #       and backward direction
    num_equations = 2 * (x_lens.sum() - x_lens.size * order)
    tikhonov_lambda = get_validated_real_numeric(
        value=tikhonov_lambda if tikhonov_lambda is not None else 0.0,
        name="tikhonov_lambda",
    )

    try:
        lstsq_solver = {  # type: ignore
            "sym": "sym",
            "symmetric": "sym",
            "pos": "pos",
            "positive_definite": "pos",
        }[lstsq_solver.lower()]

    except KeyError:
        raise ValueError(
            f"Expected 'lstsq_solver' to be 'sym', 'symmetric', 'pos', or "
            f"'positive_definite', but got '{lstsq_solver}'."
        )

    # --- Computation ---

    # depending on the choice of the user, the Numba-accelerated or the NumPy-based
    # implementation is used
    try:
        return _ar_one_step_least_squares(
            xs=xs,
            x_lens=x_lens,
            order=order,
            num_equations=num_equations,
            tikhonov_lambda=tikhonov_lambda,
            lstsq_solver=lstsq_solver,  # type: ignore
        )

    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError(
            f"The matrix inversion failed because the design matrix is singular with "
            f"the given Tikhonov regularisation parameter 'tikhonov_lambda' of "
            f"{tikhonov_lambda:.5e}.\n"
            f"Please consider\n"
            f"- reducing 'order'\n"
            f"- increasing 'tikhonov_lambda'\n"
            f"- using a different solver"
        ) from error


def extrapolate_autoregressive(
    x: ArrayLike,
    ar_coeffs: Union[ArrayLike, Tuple[ArrayLike, ArrayLike], List[ArrayLike]],
    pad_width: Union[Integer, Tuple[Integer, Integer], List[Integer]] = (0, 0),
    zero_lag_warn: bool = True,
) -> NDArray[np.float64]:
    """
    Extrapolates a signal beyond its original range using the coefficients of one or
    two autoregressive models.

    If available at runtime, a Numba-accelerated implementation is used instead of the
    NumPy-based one.

    Parameters
    ----------
    x : Array-like of shape (n,)
        The real input signal to be extrapolated.
        It is internally promoted to ``numpy.float64``.
        Its length has to be at least ``2``.
    ar_coeffs : Array-like of shape (order + 1,) or 2-tuple or 2-list of Array-likes with shapes (order1 + 1,) and (order2 + 1,)
        The AR coefficients of the autoregressive model(s).
        If only a single Array-like is provided, these coefficients are applied to both
        the left hand side and the right hand side extrapolation.
        For an iterable of two Array-likes, the first one is used for the left hand side
        and the second one for the right hand side extrapolation. Both models can even
        have different orders as long as there are at least ``2`` (AR(1) model) and at
        most ``len(x) + 1`` coefficients.
        Independent of the order(s), it is expected that the ``i``-th element
        corresponds to the ``i``-th lag. This implies that the zero-lag coefficient
        is present at index ``0``.
        In case the zero-lag coefficient is not exactly equal to ``1.0``, all
        coefficients of the respective model are normalised by this value and a warning
        is issued (see ``zero_lag_warn``).
        The coefficients are internally promoted to ``numpy.float64``.
    pad_width : :class:`int` or 2-tuple or 2-list of :class:`int`, default=``(0, 0)``
        The size of the extrapolation on the left and right side.
        If only a single integer is provided, the same padding is applied to both sides.
        For an iterable of two integers, the first one is used for the left hand side
        and the second one for the right hand side extrapolation.
        Negative values are silently clipped to ``0``, which means that no extrapolation
        is performed on the respective side(s).
    zero_lag_warn : :class:`bool`, default=``True``
        Whether to issue a warning if the zero-lag coefficient of the AR model is not
        exactly equal to ``1.0`` (``True``) or not (``False``).
        Setting this to ``False`` will only disable the warning and not the
        normalisation.

    Returns
    -------
    x_extrapolated : :class:`numpy.ndarray` of shape (n + pad_left + pad_right,) of dtype ``numpy.float64``
        The extrapolated signal.

    Raises
    ------
    TypeError
        If ``x``, ``ar_coeffs``, or ``pad_width`` are not of the expected type.
    ValueError
        If ``x`` or ``ar_coeffs`` are not 1D Array-like or an iterable of 1D
        Array-likes.
    ValueError
        If ``x`` or ``ar_coeffs`` are not of expected size.

    """  # noqa: E501

    # --- Input Validation ---

    x_internal = get_validated_real_numeric_1d_array_like(
        value=x,
        name="x",
        min_size=2,
        max_size=None,
        output_dtype=np.float64,
    )

    pad_width = _get_validated_pad_width(pad_width=pad_width)

    # if the padding is zero, the extrapolation is equivalent to the original signal
    # which allows for an early return to prevent unnecessary validation and computation
    if pad_width <= (0, 0):
        return x_internal

    (
        ar_coeffs_internal,
        ar_order_left,
        ar_order_right,
    ) = _prepare_ar_coeffs_for_extrapolation(
        ar_coeffs=ar_coeffs,
        signal_size=x_internal.size,
        zero_lag_warn=zero_lag_warn,
    )

    # --- Computation ---

    # the Numba-accelerated or the NumPy-based implementation is used depending on the
    # user's choice
    return _extrapolate_autoregressive(
        x=x_internal,
        ar_coeffs=ar_coeffs_internal,
        ar_order_left=ar_order_left,
        ar_order_right=ar_order_right,
        pad_width_left=pad_width[0],
        pad_width_right=pad_width[1],
    )
