"""
Module :mod:`augment.cross_fade._cross_fade_base`

This module implements weight-generating functions for cross-fading signals into each
other, like

- linear
- smooth hyperbolic tangent (sigmoidal)

Besides, it implements the cross fading of two signals using these weights.

"""

# === Setup ===

__all__ = [
    "CrossFadeWeightsCallable",
    "linear_cross_fade_weights",
    "smooth_tanh_cross_fade_weights",
    "cross_fade",
]

# === Imports ===

from functools import wraps
from typing import Protocol, Tuple

import numpy as np
from numpy.typing import NDArray

from ..._utils import (
    Integer,
    RealNumeric,
    get_validated_integer,
    get_validated_real_numeric,
    jit,
)

# === Typing ===


class CrossFadeWeightsCallable(Protocol):
    """
    Protocol for cross-fade weight generators.

    """

    def __call__(
        self,
        size: Integer,
        *args,
        **kwargs,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...


# === Auxiliary Functions ===


def _with_size_validation(
    function: CrossFadeWeightsCallable,
) -> CrossFadeWeightsCallable:
    """
    Decorator that validates the ``size`` parameter of the cross-fade weights.

    Parameters
    ----------
    function : callable
        The cross-fade weight generator function which has to have the signature
        ``func(size: int, *args, **kwargs) -> Tuple[numpy.NDArray[numpy.float64], numpy.NDArray[numpy.float64]]``.

    Returns
    -------
    function_with_size_validation : callable
        The decorated cross-fade weight generator function that handles the input
        validation of the ``size`` parameter.

    """  # noqa: E501

    @wraps(function)
    def function_with_size_validation(
        size: Integer,
        *args,
        **kwargs,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

        # --- Input Validation ---

        size = get_validated_integer(
            value=size,
            name="size",
            min_value=2,
        )

        # --- Computation ---

        return function(size, *args, **kwargs)

    return function_with_size_validation


# === Functions ===


@_with_size_validation
@jit(
    "UniTuple(float64[:], 2)(int64)",
    nopython=True,
    cache=True,
)
def linear_cross_fade_weights(
    size: Integer,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generates linear cross-fade weights.

    Parameters
    ----------
    size : :obj:`int`
        The size of the signals to cross-fade.
        It has to be at least 2.

    Returns
    -------
    weights_first, weights_second : :obj:`numpy.ndarray` of shape (size,) and dtype ``numpy.float64``
        The cross-fade weights for the first and second signal, respectively.

    Raises
    ------
    TypeError
        If ``size`` is not an integer.
    ValueError
        If ``size`` is less than 2.

    Notes
    -----
    The weights are generated as follows:

    ```
    x = i / (size - 1)
    w1[i] = 1.0 - x
    w2[i] = x
    ```

    where ``i`` is the zero-based index.

    """  # noqa: E501

    weights_first = np.linspace(1.0, 0.0, size)

    return weights_first, 1.0 - weights_first


@jit(
    "UniTuple(float64[:], 2)(int64, float64)",
    nopython=True,
    cache=True,
)
def _smooth_tanh_cross_fade_weights(
    size: int,
    slope_factor: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generates cross-fade weights using a smoothly transitioning hyperbolic tangent
    function that is C(inf)-continuous.

    Parameters
    ----------
    size : :obj:`int`
        The size of the signals to cross-fade.
        It has to be at least 2.
    slope_factor : :obj:`float`
        The slope factor of the hyperbolic tangent function.
        It may not lie below ``sqrt(3)`` because otherwise multiple inflection points
        would be introduced and the function would not be suitable for cross-fading
        anymore.
        Higher values will lead to sharper transitions happening over a smaller window
        close to the center of the cross-fade.

    Returns
    -------
    weights_first, weights_second : :obj:`numpy.ndarray` of shape (size,) and dtype `numpy.float64`
        The cross-fade weights for the first and second signal, respectively.

    """  # noqa: E501

    # the weights are computed, but the first and last weights are special set
    # manually
    # NOTE: x goes from -1 to 1, but the first and last points are excluded
    x = -1 + 2 * np.arange(1, 1 * size - 1, dtype=np.int64) / (size - 1)
    weights_second = np.empty(size, dtype=np.float64)
    weights_second[0] = 0.0
    weights_second[1 : size - 1] = 0.5 * (
        1.0 + np.tanh(slope_factor * (x / (1.0 - np.square(x))))
    )
    weights_second[size - 1] = 1.0

    return 1.0 - weights_second, weights_second


@_with_size_validation
def smooth_tanh_cross_fade_weights(
    size: Integer,
    slope_factor: RealNumeric = 1.74,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generates cross-fade weights using a smoothly transitioning hyperbolic tangent
    function that is C(inf)-continuous.

    Parameters
    ----------
    size : :obj:`int`
        The size of the signals to cross-fade.
        It has to be at least 2.
    slope_factor : :obj:`float`, default=``1.74``
        The slope factor of the hyperbolic tangent function.
        It may not lie below ``sqrt(3)`` because otherwise multiple inflection points
        would be introduced and the function would not be suitable for cross-fading
        anymore.
        Thus, the default value of ``1.74`` is only slightly above this threshold to
        ensure a smooth transition.
        Higher values will lead to sharper transitions happening over a smaller window
        close to the center of the cross-fade.

    Returns
    -------
    weights_first, weights_second : :obj:`numpy.ndarray` of shape (size,) and dtype ``numpy.float64``
        The cross-fade weights for the first and second signal, respectively.

    Raises
    ------
    TypeError
        If ``size`` or ``slope_factor`` are not of the expected type.
    ValueError
        If ``size`` is less than 2.
    ValueError
        If ``slope_factor`` is less than ``sqrt(3) = 1.7320508075688772``.


    Notes
    -----
    The weights are generated as follows:

    ```
    x = -1 + 2 * (i / (size - 1))  # from -1 to 1
    w1[i] = 1.0 - 0.5 * (1.0 + tanh(slope_factor * (x / (1.0 - x * x)))) for 1 <= i < size - 1
    w2[i] = 1.0 - w1[i]

    # special edge cases
    w1[0] = 1.0
    w2[0] = 0.0
    w1[size - 1] = 0.0
    w2[size - 1] = 1.0
    ```

    where ``i`` is the zero-based index.

    """  # noqa: E501

    # --- Input Validation ---

    slope_factor = get_validated_real_numeric(
        value=slope_factor,
        name="slope_factor",
        min_value=1.7320508075688772,  # sqrt(3)
        min_inclusive=False,
    )

    # --- Computation ---

    return _smooth_tanh_cross_fade_weights(
        size=size,  # type: ignore
        slope_factor=slope_factor,
    )


@jit(
    "float64[:](float64[:], float64[:], float64[:], float64[:])",
    nopython=True,
    cache=True,
)
def cross_fade(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    weights1: NDArray[np.float64],
    weights2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Cross-fades two signals using the given weights.

    Parameters
    ----------
    x1, x2 : :obj:`numpy.ndarray` of shape (n1,) and (n2,) and dtype ``numpy.float64``
        The signals to cross-fade. ``x1`` is cross-faded into ``x2``, i.e., ``x1`` will
        come first, transition into ``x2``, and then ``x2`` will be the only signal.
        Their size does not have to match as long as they are greater than or equal to
        the size of the weights. Please refer to the Notes section for details.
    weights1, weights2 : :obj:`numpy.ndarray` of shape (m,) and dtype ``numpy.float64``
        The cross-fade weights for the first and second signal, respectively.
        Their sizes have to match and be even. Please refer to the Notes section for
        details.

    Returns
    -------
    x_cross_faded : :obj:`numpy.ndarray` of shape (n1 + n2 - m,) and dtype ``numpy.float64``
        The cross-faded signal.
        During the cross-fading ``weights1.size`` samples have to be discarded to
        create the transitioning overlap implied by the cross-fading.

    Notes
    -----
    The cross-fading of ``x1`` into ``x2`` is done by concatenating them in the
    following way:

    ```python
    # n1 is the size of x1
    # n2 is the size of x2
    # m is the size of the weights

    # first, the cross-faded section is created from the last part of x1 and the first
    # part of x2
    cross_faded_section = weights1 * x1[-m::] + weights2 * x2[0:m]

    # then, the cross-faded section is inserted between the two signal which need to
    # have a part of their signal removed due to the overlap implied by cross-fading
    x_cross_faded = np.concatenate(
        (
            x1[0 : n1 - m],
            cross_faded_section,
            x2[m : n2],
        )
    )
    ```

    """  # noqa: E501

    cross_faded_section = (
        weights1 * x1[x1.size - weights1.size : x1.size]  # to avoid negative indices
        + weights2 * x2[0 : weights2.size]
    )

    return np.concatenate(
        (
            x1[0 : x1.size - weights1.size],
            cross_faded_section,
            x2[weights2.size : x2.size],
        )
    )
