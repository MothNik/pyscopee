"""
Mod :mod:`_utils`

This module provides utilities used across the ``pyscopee`` package.

"""

# === Imports ===

from ._custom_types import Integer, RealNumeric, RealNumericArrayLike  # noqa: F401
from ._miscellaneous import split_class_name_to_readable, warn_verbose  # noqa: F401
from ._numba_helpers import jit  # noqa: F401
from ._validate import (  # noqa: F401
    NumPyDTypeKinds,
    get_validated_integer,
    get_validated_numeric_nd_array_like,
    get_validated_real_numeric,
    get_validated_real_numeric_1d_array_like,
    get_validated_real_numeric_2d_array_like,
    isinstance_incl_none,
    validate_1d_array_is_evenly_spaced,
    validate_1d_array_is_sorted,
)
from ._visuals import (  # noqa: F401
    apply_pyscopee_plot_style,
    get_pyscopee_style,
    pyscopee_plot_style,
)
