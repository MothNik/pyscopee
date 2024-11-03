"""
Module :mod:`augment.extrapolate._numba_base`

This module implements Numba-based basic functions for extrapolating signals beyond
their original range via, e.g.,

- the Burg method for autoregressive model estimation

"""

# === Setup ===

__all__ = [
    "numba_arburg_fast",
    "numba_extrapolate_autoregressive",
]

# === Imports ===


from ..._utils import do_numba_normal_jit_action
from ._numpy_base import arburg_fast, predict_autoregressive_one_side

# === Functions ===

# NOTE: here are no functions because
#       - most NumPy-based functions from ``._numpy_base`` are compatible with
#           ``jit``-compilation already
#       - the functions that rely on ``jit``-compiled functions cannot be declared here


# === Compilation ===

# if Numba is available the functions are ``jit``-compiled
try:
    import numpy as np

    if do_numba_normal_jit_action:  # pragma: no cover
        from numba import jit
    else:
        from ..._utils import no_jit as jit

    # if enabled, the functions are compiled
    numba_arburg_fast = jit(
        "float64[:](float64[:,:], int64[:], int64, float64)",
        nopython=True,
        cache=True,
    )(arburg_fast)

    numba_predict_autoregressive_one_side = jit(
        nopython=True,
        cache=True,
    )(predict_autoregressive_one_side)

    # the function ``_numpy_base.extrapolate_autoregressive`` is now re-defined here
    # because it relies on the compiled function ``numba_predict_autoregressive_one_side``  # noqa: E501
    @jit(nopython=True, cache=True)
    def numba_extrapolate_autoregressive(
        x: np.ndarray,
        ar_coeffs: np.ndarray,
        pad_width_left: int,
        pad_width_right: int,
    ) -> np.ndarray:
        """
        Extrapolates a signal beyond its original range using the coefficients of
        an autoregressive model.

        Parameters
        ----------
        x : :class:`numpy.ndarray` of shape (n,)
            The real input signal to be extrapolated.
        ar_coeffs : :class:`numpy.ndarray` of shape (order + 1,)
            The AR coefficients of the autoregressive model.
            The zero-lag coefficient ``ar_coeffs[0]`` is expected to be present and
            exactly equal to ``1.0``.
        pad_width_left, pad_width_right : :class:`int`
            The size of the extrapolation on the left and right side of the input
            signal, respectively. Negative values are silently clipped to ``0``, which
            means that no extrapolation is performed on the respective side.

        Returns
        -------
        x_extrapolated : :class:`numpy.ndarray` of shape (n + pad_left + pad_right,)
            The extrapolated signal.

        """

        return np.concatenate(
            (
                numba_predict_autoregressive_one_side(  # type: ignore
                    x=x,  # type: ignore
                    ar_coeffs=ar_coeffs,  # type: ignore
                    pad_width=pad_width_left,  # type: ignore
                    is_left_side=True,  # type: ignore
                ),
                x,
                numba_predict_autoregressive_one_side(  # type: ignore
                    x=x,  # type: ignore
                    ar_coeffs=ar_coeffs,  # type: ignore
                    pad_width=pad_width_right,  # type: ignore
                    is_left_side=False,  # type: ignore
                ),
            )
        )


# if Numba is not available, the NumPy-based implementations are declared as the
# Numba-based implementations
except ImportError:  # pragma: no cover
    from ._numpy_base import extrapolate_autoregressive

    numba_arburg_fast = arburg_fast
    numba_extrapolate_autoregressive = extrapolate_autoregressive
