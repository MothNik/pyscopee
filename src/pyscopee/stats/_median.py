"""
Module :mod:`stats._median`

This module provides functions for estimating the median of a dataset,

- a univariate weighted robust Harrell-Davis median estimator

"""

# === Setup ===

__all__ = [
    "effective_sample_size",
    "trimmed_weighted_harrell_davis_median",
]

# === Imports ===

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.special import betainc

from pyscopee._utils import (
    RealNumericArrayLike,
    get_validated_real_numeric_1d_array_like,
    jit,
)

# === Auxiliary Functions ===


@jit(
    "float64(float64[:], boolean)",
    nopython=True,
    cache=True,
)
def _kish_effective_sample_size(
    weights: NDArray[np.float64],
    weights_are_normalised: bool,
) -> float:
    """
    Computes the Kish effective sample size from the weights of a sample.

    Parameters
    ----------
    normalised_weights : :obj:`numpy.ndarray` of shape (n,) and dtype ``numpy.float64``
        The weights of the data points.
        They can either be normalised already or not.
    weights_are_normalised : :obj:`bool`
        Whether the weights are normalised to sum up to 1 (``True``) or not (``False``).

    Returns
    -------
    effective_sample_size : :obj:`float
        The effective sample size.

    Notes
    -----
    The Kish effective sample size is defined as

    ```
    normalised_weights = weights / weights.sum()
    effective_sample_size = 1.0 / (normalised_weights ** 2).sum()
    ```

    References
    ----------
    .. [1] Akinshin A., Weighted quantile estimators, 2023, arXiv:2304.07265

    """

    if not weights_are_normalised:
        weights = weights / weights.sum()

    return float(1.0 / np.square(weights).sum())


def _get_validated_weights(
    weights: Optional[RealNumericArrayLike],
    expected_size: Optional[int],
) -> NDArray[np.float64]:
    """
    Validates the weights of the data points.

    Parameters
    ----------
    weights : ArrayLike of shape (n,) or ``None``
        The weights of the data points.
        Negative weights are silently clipped to zero.
        If ``None``, all data points are assumed to have equal weight.
        Its data type is internally promoted to ``numpy.float64``.
    expected_size : :obj:`int` or ``None``
        The expected size of the weights array.
        If ``None``, the size is not checked. However, this cannot be ``None`` if
        ``weights`` is also ``None``.

    Returns
    -------
    weights : :obj:`numpy.ndarray` of shape (n,) and dtype ``numpy.float64``
        The validated normalised weights.

    Raises
    ------
    TypeError
        If ``weights`` is not of the expected type.
    ValueError
        If explicitly provided ``weights`` are not of ``expected_size`` or empty.
    ValueError
        If ``weights`` and ``expected_size`` are both ``None``.
    ValueError
        If all ``weights`` are zero after clipping.

    """

    # for the unweighted case, all weights are set to 1.0
    if weights is None:
        if expected_size is None:
            raise ValueError(
                "Both 'weights' and 'expected_size' cannot be None at the same time."
            )

        return np.ones(
            shape=(expected_size,),
            dtype=np.float64,
        )

    # otherwise, the weights need to be validated
    weights = get_validated_real_numeric_1d_array_like(
        value=weights,
        name="weights",
        min_size=expected_size,
        max_size=expected_size,
        output_dtype=np.float64,
    )

    if (weights < 0.0).any():
        weights = np.maximum(weights, 0.0)

    if (weights == 0.0).all():
        raise ValueError(
            "All sample weights are zero after clipping negative values to zero."
        )

    return weights


# === Functions ===


def effective_sample_size(
    weights: Optional[RealNumericArrayLike] = None,
) -> float:
    """
    Computes the effective sample size of a dataset that has different weights assigned
    to each data point.

    Parameters
    ----------
    weights : ArrayLike of shape (n,) or ``None``, default=``None``
        The weights of the data points.
        Negative weights are silently clipped to zero.
        If ``None``, all data points are assumed to have equal weight.
        Its data type is internally promoted to ``numpy.float64``.

    Returns
    -------
    effective_sample_size : :obj:`float`
        The effective sample size.

    Raises
    ------
    TypeError
        If ``weights`` is not of the expected type.
    ValueError
        If explicitly provided ``weights`` are empty.
    ValueError
        If all ``weights`` are zero after clipping.

    """

    # === Input Validation ===

    weights = _get_validated_weights(
        weights=weights,
        expected_size=None,
    )

    # === Computation ===

    return _kish_effective_sample_size(
        weights=weights,
        weights_are_normalised=False,
    )


def trimmed_weighted_harrell_davis_median(
    sample: RealNumericArrayLike,
    weights: Optional[RealNumericArrayLike] = None,
) -> float:
    """
    Computes the trimmed Harrell-Davis median estimator of a dataset that has different
    weights assigned to each data point.

    It is an alternative to the standard median and allows for simple weighting while
    still providing a high breakdown point due to trimming the tails of the weighted
    sample.

    Parameters
    ----------
    sample : ArrayLike of shape (n,)
        The dataset for which the median should be estimated.
        Its data type is internally promoted to ``numpy.float64``.
    weights : ArrayLike of shape (n,) or ``None``, default=``None``
        The weights of the data points.
        Negative weights are silently clipped to zero.
        If ``None``, all data points are assumed to have equal weight.
        Its data type is internally promoted to ``numpy.float64``.

    Returns
    -------
    median : :obj:`float`
        The trimmed weighted Harrell-Davis median estimator.

    Raises
    ------
    TypeError
        If ``sample`` or ``weights`` are not of the expected type.
    ValueError
        If ``sample`` or explicitly provided ``weights`` differ in size or any of them
        is empty.
    ValueError
        If explicitly provided ``weights`` sum up to zero.

    """

    # === Input Validation ===

    sample = get_validated_real_numeric_1d_array_like(
        value=sample,
        name="sample",
        output_dtype=np.float64,
    )

    weights = _get_validated_weights(
        weights=weights,
        expected_size=sample.size,
    )

    # === Computation ===

    # first, the effective sample size is computed together with the cumulative sum
    # of the weights after sorting the sample in ascending order
    sorted_indices = np.argsort(sample)
    sample = sample[sorted_indices]
    weights = weights[sorted_indices] / weights.sum()
    effective_sample_size = _kish_effective_sample_size(
        weights=weights,  # type: ignore
        weights_are_normalised=True,
    )
    weights_cumsum = np.cumsum(np.append(np.array([0.0]), weights))  # type: ignore

    # next, the center region of maximum density of the respective beta distribution is
    # computed
    # NOTE: this center region of maximum density is located symmetrically around 0.5
    beta_param_alpha = 0.5 * (effective_sample_size + 1)
    beta_param_beta = beta_param_alpha
    density_size = 1.0 / np.sqrt(effective_sample_size)
    lower_bound = 0.5 - 0.5 * density_size
    upper_bound = 0.5 + 0.5 * density_size
    lower_bound_beta_value = betainc(beta_param_alpha, beta_param_beta, lower_bound)
    beta_scaling_factor = 1.0 / (
        betainc(beta_param_alpha, beta_param_beta, upper_bound) - lower_bound_beta_value
    )

    # only the values that are within the bounds will be used for the median
    beta_distribution_weights = np.empty_like(weights_cumsum)
    index_below_lower = np.searchsorted(
        a=weights_cumsum,
        v=lower_bound,
        side="left",
    )
    index_above_upper = np.searchsorted(
        a=weights_cumsum,
        v=upper_bound,
        side="right",
    )

    beta_distribution_weights[0:index_below_lower] = 0.0
    beta_distribution_weights[index_below_lower:index_above_upper] = (
        beta_scaling_factor
        * (
            betainc(
                beta_param_alpha,
                beta_param_beta,
                weights_cumsum[index_below_lower:index_above_upper],
            )
            - lower_bound_beta_value
        )
    )
    beta_distribution_weights[index_above_upper:] = 1.0
    diffs = np.diff(beta_distribution_weights)

    # finally, the weighted Harrell-Davis median is computed
    return float(
        np.average(
            sample,
            weights=diffs,
        )
    )
