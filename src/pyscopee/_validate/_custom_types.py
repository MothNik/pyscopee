"""
Mod :mod:`_validate._custom_types`

This module provides type definitions subject to input validation across the
``pyscopee`` package.

"""

# === Setup ===

__all__ = [
    "RealNumeric",
    "Integer",
    "RealNumericArrayLike",
]

# === Imports ===

from typing import Union

import numpy as np
from numpy.typing import ArrayLike

# === Type Definitions ===

# a real numeric value
RealNumeric = Union[int, float, np.integer, np.floating]
Integer = Union[int, np.integer]

# a real numeric Arraylike
RealNumericArrayLike = Union[RealNumeric, ArrayLike]
