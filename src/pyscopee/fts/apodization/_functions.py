"""
Module :mod:`fts.apodization._functions`

This module provides different apodization functions.

"""

# === Imports ===

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ..._utils import (
    RealNumeric,
    RealNumericArrayLike,
    get_validated_real_numeric,
    get_validated_real_numeric_1d_array_like,
)

# === Classes ===


class ApodizationFunction:
    """
    Represents an apodization function.

    Parameters
    ----------
    fun: Callable[
    """
