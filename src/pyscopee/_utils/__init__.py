"""
Mod :mod:`_utils`

This module provides utilities used across the ``pyscopee`` package.

"""

# === Imports ===

from ._custom_types import Integer, RealNumeric, RealNumericArrayLike  # noqa: F401
from ._numba_helpers import do_numba_normal_jit_action, no_jit  # noqa: F401
from ._validate import (  # noqa: F401
    get_validated_integer,
    get_validated_real_numeric,
    get_validated_real_numeric_1d_array_like,
)
