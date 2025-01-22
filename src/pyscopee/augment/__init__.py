"""
Module :mod:`augment`

This module provides functions for augmenting data, e.g.,

- extrapolating signals beyond their original range

"""

# === Imports ===

from .extrapolate import (  # noqa: F401
    ar_ordinary_least_squares,
    arburg,
    extrapolate_autoregressive,
)
