"""
Module :mod:`linalg`

This module provides linear algebra utilities in analogy to :mod:`numpy.linalg` and
:mod:`scipy.linalg`.

"""

# === Imports ===

from ._banded_linalg import (  # noqa: F401
    BandedLUFactorization,
    lu_banded,
    lu_solve_banded,
    slogdet_lu_banded,
)
