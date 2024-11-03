"""
This test suite implements all tests for the autoregressive extrapolation via the
function `pyscopee.augment.extrapolate.extrapolate_autoregressive`.

"""

# === Imports ===

import numpy as np
import pytest

from pyscopee.augment.extrapolate import extrapolate_autoregressive

# === Tests ===


@pytest.mark.parametrize("jit", [False, True])
def test_ar_extrapolation_does_nothing_for_zero_pad_width(
    jit: bool,
) -> None:
    """
    Checks that the function :func:`pyscopee.augment.extrapolate.extrapolate_autoregressive`
    does not change the input signal if the extrapolation width is zero.

    """  # noqa: E501

    np.random.seed(42)
    x = np.random.rand(100)

    x_extrapolated = extrapolate_autoregressive(
        x=x,
        ar_coeffs=np.array([1.0, 0.5, -0.3]),
        pad_width=(0, 0),
        jit=jit,
    )

    assert np.array_equal(x, x_extrapolated)

    return


@pytest.mark.parametrize("jit", [False, True])
def test_ar_extrapolation_warns_for_non_unity_zero_lag_coeff(
    jit: bool,
) -> None:
    """
    Checks that the function :func:`pyscopee.augment.extrapolate.extrapolate_autoregressive`
    issues a warning if the zero-lag coefficient is not unity.

    """  # noqa: E501

    np.random.seed(42)
    x = np.random.rand(100)

    with pytest.warns(
        RuntimeWarning,
        match=(
            "The zero-lag coefficient of the AR model is not exactly 1.0, but "
            "5.00000e-01."
        ),
    ):
        extrapolate_autoregressive(
            x=x,
            ar_coeffs=np.array([0.5, 0.5, -0.3]),
            pad_width=(10, 10),
            jit=jit,
        )

    return
