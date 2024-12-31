"""
Module :mod:`spectra_simulate.window`

This module implements a very simple window to achieve smooth cutoffs for spectral
ranges.

"""

# === Imports ===

import numpy as np
from numpy.typing import NDArray

from pyscopee._utils import (
    RealNumeric,
    RealNumericArrayLike,
    get_validated_real_numeric,
    get_validated_real_numeric_1d_array_like,
)

# === Functions ===


def window_three_segment_smooth_cutoff(
    x: RealNumericArrayLike,
    x_min1: RealNumeric,
    x_max1: RealNumeric,
    x_min2: RealNumeric,
    x_max2: RealNumeric,
    exponent1: RealNumeric = 20,
    exponent2: RealNumeric = 20,
) -> NDArray[np.float64]:
    """
    Computes a smooth window that is

    - increasing from 0 to 1 in the range ``[x_min1, x_max1]`` ("ramp-up")
    - constant at 1 in the range ``[x_max1, x_min2]`` ("plateau")
    - decreasing from 1 to 0 in the range ``[x_min2, x_max2]`` ("ramp-down")

    and has C1 continuity at the transition points.

    The window is defined as:

    ```
    w1(x) = 0.5 * (1 + cos(pi * (u1 ** m1)))
    w2(x) = 1
    w3(x) = 0.5 * (1 + cos(pi * (u2 ** m2)))

    u1 = (x_max1 - x) / (x_max1 - x_min1)
    u2 = (x - x_min2) / (x_max2 - x_min2)
    ```

    where ``m1`` and ``m2`` are the exponents that control how sharp the ramp-up and
    ramp-down are, respectively.

    Parameters
    ----------
    x : class:`int` or :class:`float` or :class:`numpy.ndarray` of shape (n,)
        The x-values for which the window function should be computed.
        Its data type is internally promoted to ``numpy.float64``.
    x_min1, x_max1, x_min2, x_max2 : class:`int` or :class:`float`
        The boundaries of the ramp-up and ramp-down regions, respectively.
        Flipped or coinciding values for the individual parts will result in an error.
    exponent1, exponent2 : class:`int` or :class:`float`, default=``20``
        The exponents ``m1`` and ``m2`` that control how sharp the ramp-up and ramp-down
        are, respectively.
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
        If the bounds of the ramp-up and ramp-down are flipped or coinciding.
    ValueError
        If the ramp-up and ramp-down bounds are not in the correct order.
    ValueError
        If the exponents are not positive.

    References
    ----------
    .. [1] Lee L., et al., Extension of the Norton–Beer apodizing functions in Fourier
       transform spectrometry, Applied Optics, Volume 6, Issue 20, pp. 4622 - 4626,
       2012, DOI: 10.1364/AO.51.004622
       
    """

    # --- Input Validation ---

    # the x-values are checked and converted to a 1D NumPy Array
    x = get_validated_real_numeric_1d_array_like(
        value=x,
        name="x",
        output_dtype=np.float64,
    )

    # the minimum and maximum x-values are checked and converted to floats
    x_min1, x_max1, x_min2, x_max2 = [
        get_validated_real_numeric(
            value=value,
            name=name,
        )
        for value, name in zip(
            (x_min1, x_max1, x_min2, x_max2),
            ("x_min1", "x_max1", "x_min2", "x_max2"),
        )
    ]

    # if any of the bounds are flipped or coinciding, an error is raised
    if x_min1 >= x_max1:
        raise ValueError(
            f"The ramp-up bounds are flipped or coinciding ({x_min1}, {x_max1})."
        )

    if x_min2 >= x_max2:
        raise ValueError(
            f"The ramp-down bounds are flipped or coinciding ({x_min2}, {x_max2})."
        )

    if x_max1 >= x_min2:
        raise ValueError(
            f"The ramp-up and ramp-down bounds are not in the correct order. Ramp-up "
            f"ends with {x_max1:.5e} but ramp-down starts with {x_min2:.5e}."
        )

    # the exponents are checked and converted to positive floats
    exponent1, exponent2 = [
        get_validated_real_numeric(
            value=exp,
            name=name,
            min_value=1.0,
        )
        for exp, name in zip((exponent1, exponent2), ("exponent1", "exponent2"))
    ]

    # --- Nested Functions ---

    def ramp(u_values, exponent):
        return 0.5 * (1 + np.cos(np.pi * (u_values**exponent)))

    # --- Computation ---

    # only the window values for the non-zero x-values are computed
    window = np.empty_like(x)
    indices = np.where(np.logical_or(x <= x_min1, x >= x_max2))[0]
    if indices.size > 0:
        window[indices] = 0.0

    # the ramp-up is computed first
    indices = np.where(np.logical_and(x > x_min1, x < x_max1))[0]
    if indices.size > 0:
        window[indices] = ramp(
            u_values=(x_max1 - x[indices]) / (x_max1 - x_min1),
            exponent=exponent1,
        )

    # the plateau is computed next
    indices = np.where(np.logical_and(x >= x_max1, x <= x_min2))[0]
    if indices.size > 0:
        window[indices] = 1.0

    # finally, the ramp-down is computed
    indices = np.where(np.logical_and(x > x_min2, x < x_max2))[0]
    if indices.size > 0:
        window[indices] = ramp(
            u_values=(x[indices] - x_min2) / (x_max2 - x_min2),
            exponent=exponent2,
        )

    return window
