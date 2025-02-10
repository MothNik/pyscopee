"""
Module :mod:`augment.cross_fade`

This module provides a variety of cross-fade functions that can be used to blend two
signals into each other.

Currently, the following methods are implemented:

- linear
- smooth hyperbolic tangent (sigmoidal)

"""

from ._cross_fade import (  # noqa: F401
    cross_fade,
    linear_cross_fade_weights,
    smooth_tanh_cross_fade_weights,
)
