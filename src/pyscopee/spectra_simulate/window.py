"""
Module :mod:`spectra_simulate.window`

This module implements a very simple window to achieve smooth cutoffs for spectral
ranges.

"""

# === Imports ===

from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .._utils import (
    RealNumeric,
    RealNumericArrayLike,
    get_validated_real_numeric,
    get_validated_real_numeric_1d_array_like,
)

# === Functions ===


def window_with_smooth_cutoff(
    x: RealNumericArrayLike,
    x_min: RealNumeric,
    x_max: RealNumeric,
    exponent: Union[RealNumeric, Tuple[RealNumeric, RealNumeric]] = 20,
) -> NDArray[np.float64]:
    """
    Computes a smooth window function that is non-zero in the range ``[x_min, x_max]``
    and zero elsewhere.
    It is defined as:

    ```
    0.5 * (1 + cos(pi * (abs(u) ** m)))
    u = (x - 0.5 * (x_min + x_max)) / (0.5 * (x_max - x_min))
    ```

    where ``m`` is the exponent that controls how sharp the cutoff is.

    Parameters
    ----------
    x : class:`int` or :class:`float` or :class:`numpy.ndarray` of shape (n,)
        The x-values for which the window function should be computed.
        Its data type is internally promoted to ``numpy.float64``.
    x_min, x_max : class:`int` or :class:`float`
        The minimum and maximum x-values of the window function beyond which the
        function is zero.
        The window is centered at ``0.5 * (x_min + x_max)`` where it reaches its
        maximum value of ``1``.
        Flipped values are automatically corrected, but coinciding values will result
        in an error.
    exponent : class:`int` or :class:`float` or (:class:`int` or :class:`float`, :class:`int` or :class:`float`), default=``20``
        The exponent(s) ``m`` that controls how sharp the cutoff is.
        Scalars will be used for both the lower and upper cutoff.
        If provided as a tuple, the first value is used for the lower cutoff and the,
        i.e., ``x < 0.5 * (x_min + x_max)``, and the second value for the upper cutoff
        ``x > 0.5 * (x_min + x_max)``.
        A higher value results in a sharper cutoff.
        It has to be a positive value ``>= 1``.
        The default value of ``20`` is already quite close to a step function.

    Returns
    -------
    window : :class:`numpy.ndarray` of shape (n,) of dtype ``numpy.float64``
        The window function values for the given x-values.
        It will be an Array even if ``x`` is a scalar.

    Raises
    ------
    ValueError
        If ``x_min`` and ``x_max`` are equal.
    ValueError
        If the ``exponent`` is not positive.

    """  # noqa: E501

    # --- Input Validation ---

    # the x-values are checked and converted to a 1D NumPy Array
    x = get_validated_real_numeric_1d_array_like(
        value=x,
        name="x",
        output_dtype=np.float64,
    )

    # the minimum and maximum x-values are checked and converted to floats
    x_min = get_validated_real_numeric(
        value=x_min,
        name="x_min",
    )
    x_max = get_validated_real_numeric(
        value=x_max,
        name="x_max",
    )

    # if the bounds are equal, an error is raised
    if x_min == x_max:
        raise ValueError("The minimum and the maximum window bounds may not be equal.")

    # if the bounds are flipped, they are automatically corrected
    if x_min > x_max:
        x_min, x_max = x_max, x_min

    # the exponent is checked and converted to a positive float
    if not isinstance(exponent, tuple):
        exponent = (exponent, exponent)

    exponent = tuple(  # type: ignore
        get_validated_real_numeric(
            value=exp,
            name=f"exponent [{iter_i}]",
            min_value=1.0,
        )
        for iter_i, exp in enumerate(exponent)
    )

    # --- Computation ---

    # only the window values for the non-zero x-values are computed
    nonzero_indices = np.logical_and(x >= x_min, x <= x_max)
    zero_indices = np.where(np.logical_not(nonzero_indices))[0]
    nonzero_indices = np.where(nonzero_indices)[0]

    if nonzero_indices.size <= 0:
        return np.zeros_like(x)

    window = np.empty_like(x)
    window[zero_indices] = 0.0

    # to distinguish between the lower and upper cutoff, the x-values are normalised
    # and the indices are split accordingly
    u_values = (x[nonzero_indices] - 0.5 * (x_min + x_max)) / (0.5 * (x_max - x_min))
    low_cutoff_indices = np.where(u_values < 0.0)[0]
    high_cutoff_indices = np.where(u_values >= 0.0)[0]
    u_values = np.abs(u_values)

    for indices, exp in zip((low_cutoff_indices, high_cutoff_indices), exponent):  # type: ignore
        window[nonzero_indices[indices]] = 0.5 * (
            1 + np.cos(np.pi * (u_values[indices] ** exp))
        )

    return window
