"""
Module mod :mod:`spectra_simulate`

This module implements functions to simulate spectra, like the spectra of the light
sources that are attenuated by atmospheric gases.

It provides:

- a Planck blackbody radiation function
- a function to calculate the transmittance of an atmospheric gas mixture

"""

# === Imports ===

from .atmospheric import atmospheric_transmittance  # noqa: F401
from .black_body import black_body_peak, black_body_spectrum  # noqa: F401
