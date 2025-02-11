"""
Module :mod:`stats`

This module provides functions for statistical analysis such as

- a univariate weighted robust Harrell-Davis median estimator

"""

# === Imports ===

from ._median import (  # noqa: F401
    effective_sample_size,
    trimmed_weighted_harrell_davis_median,
)
