"""
Mod :mod:`_utils`

This module provides utilities used across the ``pyscopee`` package.

"""

# === Imports ===

from ._miscellaneous import split_class_name_to_readable, warn_verbose  # noqa: F401
from ._numba_helpers import jit  # noqa: F401
from ._visuals import (  # noqa: F401
    apply_pyscopee_plot_style,
    get_pyscopee_style,
    pyscopee_plot_style,
)
