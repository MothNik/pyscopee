"""
Package ``pyscopee``

A Python package for Spectroscopy in Python.

"""

# === Imports ===

import os as _os

from . import augment  # noqa: F401
from ._utils import (  # noqa: F401
    Integer,
    RealNumeric,
    RealNumericArrayLike,
    apply_pyscopee_plot_style,
    get_pyscopee_style,
    get_validated_integer,
    get_validated_real_numeric,
    get_validated_real_numeric_1d_array_like,
    pyscopee_plot_style,
)
from .fts import apodization  # noqa: F401
from .fts.apodization import (  # noqa: F401
    Boxcar,
    CustomApodization,
    Triangular,
    ZeroMappedHyperbolicSine,
    as_apodization_function,
    boxcar,
    print_apodization_function_template,
    triangular,
    zero_mapped_hyperbolic_sine,
)
from .spectra_simulate import black_body_peak, black_body_spectrum  # noqa: F401

# === Package Metadata ===

_AUTHOR_FILE_PATH = _os.path.join(_os.path.dirname(__file__), "AUTHORS.txt")
_VERSION_FILE_PATH = _os.path.join(_os.path.dirname(__file__), "VERSION.txt")

with open(_AUTHOR_FILE_PATH, "r") as author_file:
    __author__ = author_file.read().strip()

with open(_VERSION_FILE_PATH, "r") as version_file:
    __version__ = version_file.read().strip()
