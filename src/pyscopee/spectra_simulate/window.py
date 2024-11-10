"""
Module :mod:`spectra_simulate.window`

This module implements a very simple window to achieve smooth cutoffs for spectral
ranges.

"""

# === Imports ===

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
    exponent: RealNumeric = 20,
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
        Flipped values are automatically corrected, but coinciding values will result
        in an error.
    exponent : class:`int` or :class:`float`, default=``20``
        The exponent that controls how sharp the cutoff is.
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

    """

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
    exponent = get_validated_real_numeric(
        value=exponent,
        name="exponent",
        min_value=1.0,
    )

    # --- Computation ---

    # only the window values for the non-zero x-values are computed
    x_nonzero_indices = np.logical_and(x >= x_min, x <= x_max)
    x_zero_indices = np.where(np.logical_not(x_nonzero_indices))[0]
    x_nonzero_indices = np.where(x_nonzero_indices)[0]

    if x_nonzero_indices.size <= 0:
        return np.zeros_like(x)

    window = np.empty_like(x)
    window[x_zero_indices] = 0.0

    u_values = np.abs(
        (x[x_nonzero_indices] - 0.5 * (x_min + x_max)) / (0.5 * (x_max - x_min))
    )
    window[x_nonzero_indices] = 0.5 * (1 + np.cos(np.pi * (u_values**exponent)))

    return window
