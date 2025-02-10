"""
Module :mod:`augment`

This module provides functions for augmenting data, e.g.,

- extrapolating signals beyond their original range
- cross-fading (blending) signals into each other

"""

# === Imports ===

from .extrapolate import (  # noqa: F401
    ar_ordinary_least_squares,
    arburg,
    extend_grid_points,
    extrapolate_autoregressive,
)
from .cross_fade import (  # noqa: F401
    cross_fade,
    linear_cross_fade_weights,
    smooth_tanh_cross_fade_weights,
)
