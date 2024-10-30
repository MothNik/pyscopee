"""
Module :mod:`spectra_simulate.atmospheric`

This module offers functions to generate spectra of atmospheric gases like water vapour
and carbon dioxide.

It relies on the HITRAN database for the spectral data and accesses them via the
optional dependency ``radis``.

"""

# === Imports ===

import warnings

import numpy as np
from scipy.interpolate import splev, splrep

from .._utils import (
    RealNumeric,
    RealNumericArrayLike,
    get_validated_real_numeric_1d_array_like,
)
from ._validated_conversion import (
    GasConcentrationUnit,
    PathlengthUnit,
    PressureUnit,
    TemperatureUnit,
    get_checked_path_length_centimeters,
    get_checked_pressure_bar,
    get_checked_temperature_kelvin,
    get_validated_gas_mole_fraction,
)

# === Functions ===


def calc_atmospheric_transmittance(
    wavenumbers: RealNumericArrayLike,
    temperature: RealNumeric = 25.0,
    temperature_unit: TemperatureUnit = "C",
    pressure: RealNumeric = 1.0,
    pressure_unit: PressureUnit = "atm",
    h2o_mole_fraction: RealNumeric = 20_000.0,
    h2o_unit: GasConcentrationUnit = "ppmv",
    co2_mole_fraction: RealNumeric = 420.0,
    co2_unit: GasConcentrationUnit = "ppmv",
    path_length: RealNumeric = 25.0,
    path_length_unit: PathlengthUnit = "cm",
    verbose: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Calculates the atmospheric transmittance for the given wavenumbers assuming that
    the atmosphere consists of air, water vapour, and carbon dioxide.

    Parameters
    ----------
    wavenumbers : class:`int` or :class:`float` or :class:`numpy.ndarray` of shape (n,)
        The wavenumber(s) in 1 / cm.
    temperature : class:`int` or :class:`float`, default=``25.0``
        The temperature given in the specified ``temperature_unit``.
    temperature_unit : {``"K"``, ``"k"``, ``"C"``, ``"c"``, ``"F"``, ``"f"``}, default=``"K"``
        The unit of the ``temperature`` which can be:

        - ``"K"`` or ``"k"``: Kelvin
        - ``"C"`` or ``"c"``: degree Celsius
        - ``"F"`` or ``"f"``: degree Fahrenheit

    pressure : class:`int` or :class:`float`, default=``1.0``
        The pressure given in the specified ``pressure_unit``.
    pressure_unit : {``"bar"``, ``"mbar"``, ``"hPa"``, ``"kPa"``, ``"atm"``}, default=``"atm"``
        The unit of the ``pressure`` which can be:

        - ``"bar"``: bar
        - ``"mbar"``: millibar
        - ``"hPa"``: hectopascal
        - ``"kPa"``: kilopascal
        - ``"atm"``: atmospheres

    h2o_mole_fraction, co2_mole_fraction : class:`int` or :class:`float`, default=(``20_000.0``, ``420.0``)
        The mole fraction of water vapour and carbon dioxide given in the specified
        ``h2o_unit`` and ``co2_unit``, respectively.
    h2o_unit, co2_unit : {``"ppmv"``, ``"percent"``, ``"%"``}, default=``"ppmv"``
        The unit of the mole fraction which can be:

        - ``"ppmv"``: parts per million by volume
        - ``"percent"`` or ``"%"``: percent

    path_length : class:`int` or :class:`float`, default=``25.0``
        The path length given in the specified ``path_length_unit``.
    path_length_unit : {``"m"``, ``"dm"``, ``"cm"``, ``"mm"``, ``"µm"``}, default=``"cm"``
        The unit of the ``path_length`` which can be:

        - ``"m"``: meter
        - ``"dm"``: decimeter
        - ``"cm"``: centimeter
        - ``"mm"``: millimeter
        - ``"µm"``: micrometer

    verbose : :class:`bool`, default=``False``
        Whether to print additional information (``True``) or not (``False``).
    kwargs : :class:`dict`
        Additional keyword arguments to be passed to the :func:`radis.calc_spectrum`
        function.

    Returns
    -------
    atmospheric_transmittance : :class:`numpy.ndarray` of shape (1,) or (n,)
        The atmospheric transmittance for the given wavenumbers as a dimensionless
        quantity from ``0.0`` to ``1.0``.
        It will be an Array even if ``wavenumbers`` was a scalar.

    Raises
    ------
    TypeError
        If the input arguments have invalid types.
    ValueError
        If the input arguments have invalid values.
    ImportError
        If the optional dependency ``radis`` is not installed.

    """  # noqa: E501

    # --- Imports ---

    try:
        from radis import calc_spectrum
        from radis import config as radis_config
        from radis.misc.warning import AccuracyWarning

    except ImportError as error:
        raise ImportError(
            "The optional dependency 'radis' is required for this function, but it is "
            "not installed. You can install it via 'pip install radis'."
        ) from error

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

    # afterwards, the pressure is converted to bar
    pressure = get_checked_pressure_bar(
        pressure=pressure,
        pressure_unit=pressure_unit,
    )

    # the mole fractions of water vapour and carbon dioxide are converted to
    # dimensionless values
    h2o_mole_fraction = get_validated_gas_mole_fraction(
        mole_fraction=h2o_mole_fraction,
        mole_fraction_unit=h2o_unit,
    )
    co2_mole_fraction = get_validated_gas_mole_fraction(
        mole_fraction=co2_mole_fraction,
        mole_fraction_unit=co2_unit,
    )

    # finally, the path length is converted to centimeters
    path_length = get_checked_path_length_centimeters(
        path_length=path_length,
        path_length_unit=path_length_unit,
    )

    # --- Computation ---

    # the atmospheric transmittance is calculated using the RADIS package
    radis_config["MISSING_BROAD_COEF"] = "air"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=AccuracyWarning)

        with np.errstate(over="ignore", under="ignore"):
            spectrum = calc_spectrum(
                wmin=wavenumbers.min(),  # is 1 / cm if not specified
                wmax=wavenumbers.max(),
                mole_fraction={  # type: ignore
                    "H2O": h2o_mole_fraction,
                    "CO2": co2_mole_fraction,
                },
                isotope={  # type: ignore
                    "H2O": "1",
                    "CO2": "1",
                },
                pressure=pressure,
                Tgas=temperature,
                path_length=path_length,  # type: ignore
                medium="air",
                databank="hitran",
                wstep="auto",  # type: ignore
                cutoff=0.0,
                verbose=verbose,
                warnings={
                    "MissingDiluentBroadeningWarning": "ignore",
                    "MissingDiluentBroadeningTdepWarning": "ignore",
                },
                **kwargs,
            )

    # the transmittance is interpolated to the given wavenumbers
    wavenumbers_radis, transmittance_radis = spectrum.get(
        var="transmittance_noslit",
        wunit="cm-1",
    )
    tck = splrep(
        wavenumbers_radis,
        transmittance_radis,
        k=3,
        s=0.0,
    )

    return splev(  # type: ignore
        x=wavenumbers,
        tck=tck,
    )
