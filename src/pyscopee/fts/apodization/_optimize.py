"""
Module :mod:`fts.apodization._optimize`

This module provides functions for optimizing apodization functions, i.e., finding
the optimal parameters that minimize the side lobes of the Fourier transform of a
custom apodization function.

"""

# === Imports ===

import numpy as np
from numpy.typing import ArrayLike

from scipy.optimize import dual_annealing