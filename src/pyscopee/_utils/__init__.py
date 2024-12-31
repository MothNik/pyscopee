"""
Mod :mod:`_utils`

This module provides utilities used across the ``pyscopee`` package.

"""

# === Imports ===

from ._custom_types import Integer, RealNumeric, RealNumericArrayLike  # noqa: F401
from ._miscellaneous import split_class_name_to_readable  # noqa: F401
from ._numba_helpers import jit  # noqa: F401
from ._validate import (  # noqa: F401
    get_validated_integer,
    get_validated_real_numeric,
    get_validated_real_numeric_1d_array_like,
)
from ._visuals import apply_pyscopee_plot_style, get_pyscopee_style  # noqa: F401
