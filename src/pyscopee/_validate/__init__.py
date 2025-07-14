"""
Mod :mod:`_validate`

``pyscopee``'s validation module for a wide variety of common input data, such as

- numerical data types
- Array-like data structures

"""

# === Imports ===

from ._arrays import (  # noqa: F401
    get_validated_numeric_nd_array_like,
    get_validated_real_numeric_1d_array_like,
    get_validated_real_numeric_2d_array_like,
    validate_1d_array_is_evenly_spaced,
    validate_1d_array_is_sorted,
)
from ._custom_types import Integer, RealNumeric, RealNumericArrayLike  # noqa: F401
from ._numbers import get_validated_integer, get_validated_real_numeric  # noqa: F401
