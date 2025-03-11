"""
Module :mod:`ft_spectroscopy._symmetrize`

This module provides functions for symmetrizing interferograms, e.g.,

- by applying a digitial all-pass filter to it

"""

# === Imports ===

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, dual_annealing
from scipy.signal import sosfilt, sosfilt_zi

# === Auxiliary Functions ===


def _make_allpass_correction_bounds(
    num_poles: int,
    max_modulus: float,
    max_shift: float,
) -> Bounds:
    """
    Sets up the bounds for the optimisation of the poles of an all-pass filter and the
    shift of the signal.
    Please refer to the Notes section for further details.

    Parameters
    ----------
    num_poles : :obj:`int`
        The number of poles in the all-pass filter.
        It must be a positive integer ``> 0``.
    max_modulus : :obj:`float`
        The maximum modulus of the poles of the all-pass filter to ensure stability.
        It must be a positive float value in the interval ``(0, 1)`` (both exclusive).
    max_shift : :obj:`float`
        The maximum shift by which the signal is shifted after filtering it with the
        all-pass filter.
        It has to be a non-negative float value ``>= 0.0``.

    Returns
    -------
    bounds : :obj:`scipy.optimize.Bounds`
        The bounds for the optimisation of the poles of the all-pass filter and the
        shift of the signal.

    Notes
    -----
    The poles of an all-pass filter are complex numbers. Thus, they have two variables,
    namely their real and imaginary parts. However, the poles of stable all-pass filters
    lie within the unit circle, i.e., ``abs(pole) < max_modulus``. To efficiently impose
    this constraint, the poles are parametrised in polar coordinates, i.e.,
    ``pole = modulus * exp(1j * angle)``. With this, one pole is represented by two
    variables, namely its modulus and angle.

    Yet, the denominator and numerator of the all-pass filter both have real polynomials
    coefficients, i.e., the poles appear in conjugate pairs. Consequently, only half of
    the poles need to be optimised.

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

    Finally, the bound for the signal shift is added to the end of the bounds as one
    additional variable which yields

    ``[(modulus_0), modulus_1, angle_1, modulus_2, angle_2, ..., shift]``

    """

    num_purely_real_poles = num_poles % 2

    lower_bounds = np.empty(shape=(num_poles + 1,), dtype=np.float64)
    upper_bounds = np.empty_like(lower_bounds)

    # NOTE: the indexing for the purely real poles will not take any effect if there is
    #       no purely real pole
    lower_bounds[0:num_purely_real_poles] = -max_modulus
    lower_bounds[num_purely_real_poles:num_poles:2] = -max_modulus
    lower_bounds[num_purely_real_poles + 1 : num_poles : 2] = 0.0
    lower_bounds[num_poles] = 0.0

    upper_bounds[0:num_purely_real_poles] = max_modulus
    upper_bounds[num_purely_real_poles:num_poles:2] = max_modulus
    upper_bounds[num_purely_real_poles + 1 : num_poles : 2] = np.pi
    upper_bounds[num_poles] = max_shift

    return Bounds(
        lb=lower_bounds,  # type: ignore
        ub=upper_bounds,  # type: ignore
    )


def _convert_allpass_poles_to_sos(
    pole_params: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Sets up the second-order sections (SOS) of an all-pass filter given its poles.

    Parameters
    ----------
    pole_params : :obj:`numpy.ndarray` of shape ``(num_poles,)`` and dtype ``numpy.float64``
        The parameters of the poles of the all-pass filter.
        For further details, please refer to the documentation of the function
        :func:`_calc_average_phase_error_after_allpass_correction`.

    Returns
    -------
    sos_coeffs : :obj:`numpy.ndarray` of shape ``(num_sos_coeffs, 6)`` and dtype ``numpy.float64``
        The second-order sections (SOS) of the all-pass filter.
        Its number of rows will be ``num_poles // 2 + num_poles % 2`` and each row
        contains the coefficients of the numerator and denominator polynomials of the
        corresponding second-order section.

    """  # noqa: E501

    (
        num_conjugate_pairs,  # second order filters
        num_purely_real_poles,  # first order filter (if any)
    ) = divmod(pole_params.size, 2)
    sos_coeffs = np.empty(
        shape=(num_conjugate_pairs + num_purely_real_poles, 6),
        dtype=np.float64,
    )

    # NOTE: this loop will not be entered if there are no purely real poles
    for index in range(0, num_purely_real_poles):
        modulus = pole_params[index]
        # A(z) = 1.0 + a1 * inv(z) = 1.0 + modulus * inv(z)
        # B(z) = b0 + 1.0 * inv(z) = modulus + 1.0 * inv(z)
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

    for index in range(0, num_conjugate_pairs):
        modulus, angle = pole_params[
            num_purely_real_poles + 2 * index : num_purely_real_poles + 2 * index + 2
        ]
        # A(z) = 1.0 + a1 * inv(z) + a2 * inv(z)^2
        #      = 1.0 - 2.0 * modulus * cos(angle) * inv(z) + modulus^2 * inv(z)^2
        # B(z) = b0 + b1 * inv(z) + b2 * inv(z)^2
        #      = modulus^2 - 2.0 * modulus * cos(angle) * inv(z) + 1.0 * inv(z)^2
        sos_coeffs[num_purely_real_poles + index, ::] = np.array(
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


def _calc_average_phase_error_after_allpass_correction(
    correction_params: NDArray[np.float64],
    signal: NDArray[np.float64],
    pre_shift: int,
    frequency_indices: slice,
    angular_frequencies: NDArray[np.float64],
) -> float:
    """
    Calculates the average phase error of a signal after applying an all-pass filter
    correction that involves filtering the signal with an all-pass filter and shifting
    it.

    Parameters
    ----------
    correction_params : :obj:`numpy.ndarray` of shape ``(num_poles + 1,)`` and dtype ``numpy.float64``
        The parameters of the all-pass filter correction.
        Except for the last element, its elements encode the parameters of the poles of
        the all-pass filter in the form ``[(modulus_0), modulus_1, angle_1, modulus_2, angle_2, ...]``
        for the polar representation of the poles ``pole_i = modulus_i * exp(1j * angle_i)``.
        An even number of poles will result in only complex conjugate pairs of poles
        while an odd number of poles will result in pairs of complex conjugate poles
        and one additional purely real pole.
        The last element encodes the shift by which the signal is shifted after
        filtering it with the all-pass filter.
    signal : :obj:`numpy.ndarray` of shape ``(num_samples,)`` and dtype ``numpy.float64``
        The signal to be corrected.
    pre_shift : :obj:`int`
        The pre-shift by which the signal is shifted after filtering it with the
        all-pass filter and before applying the shift given by ``correction_params[-1]``.
        Note that it can only be a negative or positive integer value while
        ``correction_params[-1]`` has to be a positive float value.
        This can be helpful in case the all-pass filter and the shift have to work on
        differently centered signals.
    frequency_indices : :obj:`slice`
        The indices of the frequencies at which the phase error is calculated in the
        frequency domain.
    angular_frequencies : :obj:`numpy.ndarray` of shape (num_samples,) and dtype ``numpy.float64``
        The angular frequencies of the signal corresponding to ``frequency_indices``.

    Returns
    -------
    average_phase_error : :obj:`float`
        The average phase error of the signal after applying the all-pass filter
        correction.

    """  # noqa: E501

    pole_params = correction_params[0 : correction_params.size - 1]
    shift = correction_params[correction_params.size - 1]
    sos = _convert_allpass_poles_to_sos(pole_params=pole_params)

    signal, _ = sosfilt(
        sos=sos,
        x=signal,
        zi=sosfilt_zi(sos=sos),
    )

    filtered_fft = np.fft.rfft(
        np.roll(
            signal,
            shift=pre_shift,
        ),
    )[frequency_indices]

    # the shift is applied in the frequency domain because the FFT had to be calculated
    # anyway
    filtered_fft *= np.exp(-1.0j * angular_frequencies * shift)

    return np.mean(np.abs(np.angle(filtered_fft)))


# np.random.seed(0)
from matplotlib import pyplot as plt
from scipy.signal import sos2zpk, sosfreqz

NUM_POLES = 200

fig, ax = plt.subplots(nrows=2)

for _ in range(100):
    bounds = _make_allpass_correction_bounds(NUM_POLES, 0.995, 0.5)
    poles = np.array(
        [
            np.random.uniform(bounds.lb[index], bounds.ub[index])
            for index in range(0, len(bounds.lb) - 1)
        ]
    )
    sos = _convert_allpass_poles_to_sos(poles)

    z, p, k = sos2zpk(sos)
    print(np.abs(p))
    assert np.all(np.abs(p) <= 1.0)

    w, h = sosfreqz(sos, worN=8000)

    ax[0].plot(w, np.abs(h))

    ax[1].plot(w, np.unwrap(np.angle(h)))

ax[1].scatter(0, 0, color="red", marker="x")
ax[1].scatter(np.pi, -NUM_POLES * np.pi, color="red", marker="x")

plt.show()
