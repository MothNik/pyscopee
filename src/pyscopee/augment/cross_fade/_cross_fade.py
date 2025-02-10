"""
Module :mod:`augment.cross_fade._cross_fade`

This module provides higher level interfaces for cross-fading signals into each other.

"""

# === Setup ===

__all__ = [
    "cross_fade",
]

# === Imports ===

from typing import Any, Dict, Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..._utils import (
    Integer,
    get_validated_integer,
    get_validated_real_numeric_1d_array_like,
)
from ._cross_fade_base import CrossFadeWeightsCallable as _CrossFadeWeightsCallable
from ._cross_fade_base import cross_fade as _cross_fade
from ._cross_fade_base import linear_cross_fade_weights, smooth_tanh_cross_fade_weights

# === Constants ===

# the mapping for the supported cross-fade functions
_PYSCOPEE_CROSS_FADE_FUNCTIONS = dict(
    linear=linear_cross_fade_weights,
    smooth_tanh=smooth_tanh_cross_fade_weights,
)

# === Auxiliary Functions ===


def _get_validated_cross_fade_function(
    cross_fade_function: Union[
        Literal["linear", "smooth_tanh"],
        _CrossFadeWeightsCallable,
    ],
) -> _CrossFadeWeightsCallable:
    """
    Checks if the given cross-fade function is valid and returns the corresponding
    callable.
    Callables are not checked for their signature to avoid unnecessary overhead. Any
    failure will be caught when the callable is called.

    Parameters
    ----------
    cross_fade_function : {``"linear"``, ``"smooth_tanh"``} or callable
        The cross-fade function to validate.

    Returns
    -------
    cross_fade_function : callable
        The validated cross-fade function.

    Raises
    ------
    ValueError
        If the cross-fade function is not supported.

    """

    # if a string was provided, the respective cross-fade function is retrieved
    if isinstance(cross_fade_function, str):
        try:
            return _PYSCOPEE_CROSS_FADE_FUNCTIONS[cross_fade_function.lower()]

        except KeyError:
            available_functions = ", ".join(
                [f"'{function}'" for function in _PYSCOPEE_CROSS_FADE_FUNCTIONS.keys()]
            )
            raise ValueError(
                f"The cross-fade function '{cross_fade_function}' is not supported. "
                f"Available functions are {available_functions}."
            )

    # callables are not checked for their signature to avoid unnecessary overhead
    if callable(cross_fade_function):
        return cross_fade_function

    raise ValueError(
        f"Expected a cross-fade function as a string or a callable, but got "
        f"'{cross_fade_function}' instead."
    )


# === Functions ===


def cross_fade(
    x1: ArrayLike,
    x2: ArrayLike,
    overlap_window_size: Integer,
    cross_fade_function: Union[
        Literal["linear", "smooth_tanh"],
        _CrossFadeWeightsCallable,
    ],
    cross_fade_kwargs: Optional[Dict[str, Any]] = None,
) -> NDArray[np.float64]:
    """
    Cross-fades two signals into each other using the given cross-fade function.

    Parameters
    ----------
    x1, x2 : Array-like of shape of (n1,) and (n2,)
        The signals to cross-fade. ``x1`` is cross-faded into ``x2``, i.e., ``x1`` will
        come first, transition into ``x2``, and then ``x2`` will be the only signal.
        Their size does not have to match as long as they are within the interval
        ``[2, overlap_window_size]``.
        Please refer to the Notes section for details.
    overlap_window_size : :obj:`int`
        The size of the overlap window. In the cross-faded signal
        ``overlap_window_size`` data points will be discarded to create the overlap
        implied by the cross-fading.
        It must be ``>= 2``.
        To ensure even cross-fading, it must be an even number. Odd numbers are silently
        decreased to the next smaller even number.
    cross_fade_function : {``"linear"``, ``"smooth_tanh"``} or callable
        The cross-fade function to use.
        Available functions are:

        - ``"linear"``: a linear cross-fade function (C(0)-continuous)
        - ``"smooth_tanh"``: a smooth hyperbolic tangent cross-fade function
            (C(inf)-continuous)

        If provided as a callable, it must have the signature
        ``weights1, weights2 = cross_fade_function(size: int, **cross_fade_kwargs)``
        where ``overlap_window_size`` is passed as ``size`` together with the provided
        ``cross_fade_kwargs``.
        It is recommended to have a type checker like ``pylance`` installed which will
        warn if a callable does not match the expected signature.

    cross_fade_kwargs : {:obj:`str` : any} or ``None``, default=``None``
        Additional keyword arguments to pass to ``cross_fade_function``.

    Returns
    -------
    x_cross_faded : :obj:`numpy.ndarray` of shape (n1 + n2 - overlap_window_size,) and dtype ``numpy.float64``
        The cross-faded signal.
        During the cross-fading ``overlap_window_size`` samples have to be discarded to
        create the transitioning overlap implied by the cross-fading.

    Raises
    ------
    TypeError
        If ``x1``, ``x2``, or ``overlap_window_size`` are not of the expected type.
    ValueError
        If ``x1`` or ``x2`` are not not a real numeric 1D Array-likes with the expected
        size to support ``overlap_window_size``.
    ValueError
        If ``overlap_window_size`` is less than 2.
    ValueError
        If a callable ``cross_fade_function`` cannot be called with the expected
        signature.

    Notes
    -----
    The cross-fading of ``x1`` into ``x2`` is done by concatenating them in the
    following way:

    ```python
    # n1 is the size of x1
    # n2 is the size of x2

    # first, the cross-fading weights are computed
    weights1, weights2 = cross_fade_function(
        size=overlap_window_size,
        **cross_fade_kwargs,
    )

    # then, the cross-faded section is created from the last part of x1 and the first
    # part of x2
    cross_faded_section = (
        weights1 * x1[-overlap_window_size::]
        + weights2 * x2[0:overlap_window_size]
    )

    # finally, the cross-faded section is inserted between the two signal which need to
    # have a part of their signal removed due to the overlap implied by cross-fading
    x_cross_faded = np.concatenate(
        (
            x1[0 : n1 - overlap_window_size],
            cross_faded_section,
            x2[overlap_window_size : n2],
        )
    )
    ```

    """  # noqa: E501

    # --- Input Validation ---

    # the overlap window size is validated first because its value is needed to validate
    # ``x1`` and ``x2``
    overlap_window_size = get_validated_integer(
        value=overlap_window_size,
        name="overlap_window_size",
        min_value=2,
    )

    # if the overlap window size is not even, it is silently decreased by 1
    if overlap_window_size % 2 != 0:
        overlap_window_size -= 1

    # now, the signals can be validated
    x1, x2 = [
        get_validated_real_numeric_1d_array_like(
            value=x,
            name=f"x{index + 1}",
            min_size=overlap_window_size,
            output_dtype=np.float64,
        )
        for index, x in enumerate((x1, x2))
    ]

    # finally, the cross-fade function is validated
    cross_fade_function = _get_validated_cross_fade_function(
        cross_fade_function=cross_fade_function
    )

    if cross_fade_kwargs is None:
        cross_fade_kwargs = dict()

    # --- Computation ---

    # first, the weights are computed
    try:
        weights1, weights2 = cross_fade_function(
            size=overlap_window_size,
            **cross_fade_kwargs,
        )

    except Exception as error:
        raise ValueError(
            f"The cross-fade function '{cross_fade_function.__name__}' "  # type: ignore
            f"could not be called with the expected signature.\n"
            f"'weights1, weights2 = {cross_fade_function.__name__}"  # type: ignore
            f"(size=overlap_window_size, **cross_fade_kwargs)'\n"
            f"Error: {error}"
        )

    # then, the cross-faded section is created
    return _cross_fade(
        x1=x1,
        x2=x2,
        weights1=weights1,
        weights2=weights2,
    )
