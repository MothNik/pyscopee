"""
Mod :mod:`spectra_simulate.black_body`

This module implements a Planck blackbody radiation function with the wavenumbers
as the independent variable.

"""

# === Imports ===

from typing import Tuple

import numpy as np

from .._utils import (
    RealNumeric,
    RealNumericArrayLike,
    get_validated_real_numeric_1d_array_like,
)
from ._validated_conversion import TemperatureUnit, get_checked_temperature_kelvin

# === Constants ===

# the speed of light in m / s
_SPEED_OF_LIGHT = 299_792_45_8

# the Planck constant in J * s
_PLANCK_CONSTANT = 6.626_070_150e-34

# the Boltzmann constant in J / K
_BOLTZMANN_CONSTANT = 1.380_649e-23

# the peak wavenumber slope
_PEAK_WAVENUMBER_SLOPE = 2.821439372  # dimensionless


# === Functions ===


def black_body_spectrum(
    wavenumbers: RealNumericArrayLike,
    temperature: RealNumeric,
    temperature_unit: TemperatureUnit = "K",
) -> np.ndarray:
    """
    Computes the Planck blackbody radiation spectrum for a given temperature and
    wavenumbers.

    Parameters
    ----------
    wavenumbers : class:`int` or :class:`float` or :class:`numpy.ndarray` of shape (n,)
        The wavenumber(s) in 1 / cm.
    temperature : class:`int` or :class:`float`
        The temperature given in the specified ``temperature_unit``.
    temperature_unit : {``"K"``, ``"k"``, ``"C"``, ``"c"``, ``"F"``, ``"f"``}, default=``"K"``
        The unit of the ``temperature`` which can be:

        - ``"K"`` or ``"k"``: Kelvin
        - ``"C"`` or ``"c"``: degree Celsius
        - ``"F"`` or ``"f"``: degree Fahrenheit

    Returns
    -------
    black_body_spectrum : :class:`numpy.ndarray` of shape (1,) or (n,)
        The blackbody radiation spectrum in W * cm / (m² * sr) evaluated at the
        specified ``wavenumbers``.
        It will be an Array even if ``wavenumbers`` was a scalar.

    Raises
    ------
    ValueError
        If the ``temperature_unit`` is not one of the allowed values.
    ValueError
        If the ``temperature`` is below absolute zero after conversion to Kelvin.

    References
    ----------
    The equation was taken from [1]_ and [2]_.

    .. [1] Wikipedia, Planck's Law - Different forms,
       URL: en.wikipedia.org/wiki/Planck%27s_law#Different_forms
    .. [2] SpectralCalc.com, Calculation of Blackbody Radiance,
       URL: www.spectralcalc.com/blackbody/blackbody.html

    """  # noqa: E501

    # --- Input Validation ---

    # the wavenumbers are checked and converted to a 1D NumPy array
    wavenumbers = get_validated_real_numeric_1d_array_like(
        value=wavenumbers,
        name="wavenumbers",
    )

    # then, the temperature is converted to Kelvin
    temperature = get_checked_temperature_kelvin(
        temperature=temperature,
        temperature_unit=temperature_unit,
    )

    # --- Computation ---

    return (
        (2e8 * _PLANCK_CONSTANT * _SPEED_OF_LIGHT * _SPEED_OF_LIGHT)
        * wavenumbers
        * wavenumbers
        * wavenumbers
    ) / (
        np.exp(
            (
                (_PLANCK_CONSTANT * _SPEED_OF_LIGHT * 100.0)
                / (_BOLTZMANN_CONSTANT * temperature)
            )
            * wavenumbers
        )
        - 1.0
    )


def black_body_peak(
    temperature: RealNumeric,
    temperature_unit: TemperatureUnit = "K",
) -> Tuple[float, float]:
    """
    Computes the wavenumber at which the Planck blackbody radiation spectrum peaks
    for a given temperature as well as the peak intensity.

    Parameters
    ----------
    temperature : class:`int` or :class:`float`
        The temperature given in the specified ``temperature_unit``.
    temperature_unit : {``"K"``, ``"k"``, ``"C"``, ``"c"``, ``"F"``, ``"f"``}, default=``"K"``
        The unit of the ``temperature`` which can be:

        - ``"K"`` or ``"k"``: Kelvin
        - ``"C"`` or ``"c"``: degree Celsius
        - ``"F"`` or ``"f"``: degree Fahrenheit

    Returns
    -------
    black_body_peak_wavenumber : :class:`float`
        The wavenumber in 1 / cm at which the blackbody radiation spectrum peaks.
    black_body_peak_intensity : :class:`float`
        The peak intensity in W * cm / (m² * sr) of the blackbody radiation spectrum.

    Raises
    ------
    ValueError
        If the ``temperature_unit`` is not one of the allowed values.
    ValueError
        If the ``temperature`` is below absolute zero after conversion to Kelvin.

    References
    ----------
    The equation was taken from [1]_ and [2]_.

    .. [1] Wikipedia, Planck's Law - Different forms,
       URL: en.wikipedia.org/wiki/Planck%27s_law#Different_forms
    .. [2] SpectralCalc.com, Calculation of Blackbody Radiance,
       URL: www.spectralcalc.com/blackbody/blackbody.html

    """  # noqa: E501

    # --- Input Validation ---

    # the temperature is converted to Kelvin
    temperature = get_checked_temperature_kelvin(
        temperature=temperature,
        temperature_unit=temperature_unit,
    )

    # --- Computation ---

    black_body_peak_wavenumber = (
        _PEAK_WAVENUMBER_SLOPE
        * _BOLTZMANN_CONSTANT
        * temperature
        / (100.0 * _PLANCK_CONSTANT * _SPEED_OF_LIGHT)
    )

    black_body_peak_intensity = black_body_spectrum(
        wavenumbers=black_body_peak_wavenumber,
        temperature=temperature,
        temperature_unit="K",
    )

    return black_body_peak_wavenumber, float(black_body_peak_intensity[0])
