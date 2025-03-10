"""
Module :mod:`ft_spectroscopy._symmetrize`

This module provides functions for symmetrizing interferograms, e.g.,

- by applying a digitial all-pass filter to it

"""

# === Imports ===

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, dual_annealing
from scipy.signal import sosfilt, sosfilt_zi, ZoomFFT

# === Auxiliary Functions ===


def _make_allpass_filter_pole_bounds(
    num_poles: int,
    max_modulus: float,
) -> Bounds:
    """
    Sets up the bounds for the optimisation of the poles of an all-pass filter.
    Please refer to the Notes section for further details.

    Parameters
    ----------
    num_poles : :obj:`int`
        The number of poles in the all-pass filter.
        It must be a positive integer ``> 0``.
        Please refer to the Notes section for further details.
    max_modulus : :obj:`float`
        The maximum modulus of the poles of the all-pass filter to ensure stability.
        It must be a positive float in the interval ``(0, 1)`` (both exclusive).
        Please refer to the Notes section for further details.

    Returns
    -------
    bounds : :obj:`scipy.optimize.Bounds`
        The bounds for the optimisation of the poles of an all-pass filter.

    Notes
    -----
    The poles of an all-pass filter are complex numbers. Thus, they have two variables,
    namely their real and imaginary parts. However, the poles of stable all-pass filters
    lie within the unit circle, i.e., ``abs(pole) < max_modulus``. To efficiently impose
    this constraint, the poles are parametrised in polar coordinates, i.e.,
    ``pole = modulus * exp(1j * angle)``. With this, one pole is represented by two
    variables, namely its modulus and angle.

    However, the denominator and numerator of the all-pass filter both have real
    polynomials coefficients, i.e., the poles appear in conjugate pairs. Consequently,
    only half of the poles need to be optimised.

    In total, this means that

    - for an even number of poles, there will be ``2 * num_poles // 2 = num_poles``
        variables to optimise
    - for an odd number of poles, there will also be
        ``2 * num_poles // 2 + 1 = num_poles`` variables to optimise because one pole
        is known to be real, i.e., its ``angle = 0``

    The variable lower and upper bounds for the poles are encoded by either

    ``[modulus_1, angle_1, modulus_2, angle_2, ...]``

    for an even number of poles or

    ``[modulus_0, modulus_1, angle_1, modulus_2, angle_2, ...]``

    for an odd number of poles. ``modulus_0`` is the modulus of the only real pole.

    """

    num_purely_real_poles = num_poles % 2

    lower_bounds = np.empty(shape=(num_poles,), dtype=np.float64)
    upper_bounds = np.empty_like(lower_bounds)

    # NOTE: the indexing for the purely real poles will not take any effect if there is
    #       no purely real pole
    lower_bounds[0:num_purely_real_poles] = -max_modulus
    lower_bounds[num_purely_real_poles::2] = -max_modulus
    lower_bounds[num_purely_real_poles + 1 :: 2] = 0.0

    upper_bounds[0:num_purely_real_poles] = max_modulus
    upper_bounds[num_purely_real_poles::2] = max_modulus
    upper_bounds[num_purely_real_poles + 1 :: 2] = np.pi

    return Bounds(
        lb=lower_bounds,  # type: ignore
        ub=np.negative(lower_bounds),  # type: ignore
    )


def _make_allpass_filter_sos(
    pole_params: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Sets up the second-order sections (SOS) of an all-pass filter given its poles.

    Parameters
    ----------
    pole_params : :obj:`numpy.ndarray` of shape ``(num_poles,)`` and dtype :obj:`numpy.float64`
        The parameters of the poles of the all-pass filter.
        For further details, please refer to the documentation of the function
        :func:`_calc_imag_fraction_after_allpass_filter`.

    Returns
    -------
    sos_coeffs : :obj:`numpy.ndarray` of shape ``(half_num_poles, 6)`` and dtype :obj:`numpy.float64`
        The second-order sections (SOS) of the all-pass filter.

    """

    half_num_poles = pole_params.size // 2
    num_purely_real_poles = half_num_poles % 2
    sos_coeffs = np.empty(shape=(half_num_poles, 6), dtype=np.float64)

    # NOTE: this loop will not be entered if there are no purely real poles
    for index in range(0, num_purely_real_poles):
        modulus = pole_params[index]
        sos_coeffs[index, ::] = np.array(
            [
                modulus,  # b0
                1.0,  # b1
                0.0,  # b2
                1.0,  # a0
                modulus,  # a1
                0.0,  # a2
            ]
        )

    for index in range(num_purely_real_poles, half_num_poles):
        modulus, angle = pole_params[index : index + 2]
        sos_coeffs[index, ::] = np.array(
            [
                modulus * modulus,  # b0
                -2.0 * modulus * np.cos(angle),  # b1
                1.0,  # b2
                1.0,  # a0
                -2.0 * modulus * np.cos(angle),  # a1
                modulus * modulus,  # a2
            ]
        )

    return sos_coeffs


def _calc_average_phase_error_after_allpass(
    pole_params: 
)




bounds = _make_allpass_filter_pole_bounds(199, 0.995)
poles = np.array(
    [
        np.random.uniform(bounds.lb[index], bounds.ub[index])
        for index in range(0, len(bounds.lb))
    ]
)
sos = _make_allpass_filter_sos(poles)

from matplotlib import pyplot as plt
from scipy.signal import sosfreqz

w, h = sosfreqz(sos, worN=8000)

fig, ax = plt.subplots(nrows=2)

ax[0].plot(w, np.abs(h))

ax[1].plot(w, np.unwrap(np.angle(h)))

plt.show()
