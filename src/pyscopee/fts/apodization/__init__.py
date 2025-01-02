"""
Module :mod:`fts.apodization`

This module provides functionalities for apodization functions, e.g.,

- different apodization functions

"""

from ._classes import (  # noqa: F401
    Boxcar,
    CustomApodization,
    Triangular,
    ZeroMappedHyperbolicSine,
)
from ._functions import (  # noqa: F401
    as_apodization_function,
    boxcar,
    print_apodization_function_template,
    triangular,
    zero_mapped_hyperbolic_sine,
)
