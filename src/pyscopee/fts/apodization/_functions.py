"""
Module :mod:`fts.apodization._functions`

This module provides the function implementations used by the apodization functions in
:mod:`fts.apodization._classes`.

The functions are so simple that Numba ``jit``-compilation is not necessary. Therefore,
the functions are just implemented as plain NumPy functions.


"""

# === Setup ===

__all__ = [
    "ApodizationFunction",
    "WrappedApodizationFunction",
    "as_apodization_function",
    "boxcar",
    "get_validated_xmax",
    "not_implemented_apodization",
    "triangular",
    "zero_mapped_hyperbolic_sine",
]

# === Imports ===


import inspect
from functools import wraps
from typing import Protocol, Union

import numpy as np
from numpy.typing import NDArray

from ..._utils import (
    RealNumeric,
    RealNumericArrayLike,
    get_validated_real_numeric,
    get_validated_real_numeric_1d_array_like,
)

# === Typing ===


class _ApodizationFunctionNoKwargs(Protocol):
    def __call__(
        self,
        x: RealNumericArrayLike,
        x_max: RealNumeric = 1.0,
        *,
        skip_validation: bool = False,
    ) -> NDArray[np.float64]: ...


class _ApodizationFunctionWithKwargs(Protocol):
    def __call__(
        self,
        x: RealNumericArrayLike,
        x_max: RealNumeric = 1.0,
        *,
        skip_validation: bool = False,
        **kwargs,
    ) -> NDArray[np.float64]: ...


ApodizationFunction = Union[
    _ApodizationFunctionNoKwargs, _ApodizationFunctionWithKwargs
]
WrappedApodizationFunction = _ApodizationFunctionWithKwargs


# === Auxiliary Functions ===


def get_validated_xmax(x_max: RealNumeric) -> float:
    """
    Validates the ``x_max`` parameter of an apodization function.

    Parameters
    ----------
    x_max : :class:`float` or :class:`int`
        The maximum value of the x-range over which the apodization function is defined.
        It must be a positive real number ``> 0``.

    Returns
    -------
    x_max_validated : :class:`float`
        The validated value of ``x_max``.

    """

    return get_validated_real_numeric(
        value=x_max,
        name="x_max",
        min_value=0.0,
        min_inclusive=False,
    )


# === Functions ===


def _convert_to_validated_apodization_function(
    apodization_function: ApodizationFunction,
) -> WrappedApodizationFunction:
    """
    A decorator that wraps a given apodization function which is only defined within the
    x-interval ``[0, 1]``, where ``0` corresponds to the center of the apodization and
    ``1`` to the right boundary where it fades out (not necessarily to zero).

    The wrapper

    - symmetrizes the apodization function to the interval ``[-1, 1]`` by exploiting the
        even symmetry of apodization functions, i.e., ``f(-x) = f(x)``,
    - scales the interval ``[-1, 1]`` to ``[-x_max, x_max]``,

    x-values outside the interval ``[-x_max, x_max]`` are handled naturally by not even
    evaluating the apodization function at those points.

    For this purpose, the apodization function is expected to have the following
    signature:

        ```python
        def apodization_function(
            x: RealNumericArrayLike,
            x_max: RealNumeric,
            *,  # <- mandatory enforcement of keyword-only arguments
            ...  # <- additional keyword arguments required by the apodization function
            skip_validation: bool = False,
        ) -> NDArray[np.float64]:
            ...
        ```

    So for example, the ``boxcar`` function would be defined as:

        ```python
        def boxcar(
            x: RealNumericArrayLike,
            x_max: RealNumeric = 1.0,
            *,
            skip_validation: bool = False,
        ) -> NDArray[np.float64]:
            ...
        ```

    while the ``zero_mapped_hyperbolic_sine`` that takes the additional argument
    ``exponent`` would be defined as:

        ```python
        def zero_mapped_hyperbolic_sine(
            x: RealNumericArrayLike,
            x_max: RealNumeric = 1.0,
            *,
            exponent: RealNumeric = 5.0,
            skip_validation: bool = False,
        ) -> NDArray[np.float64]:
            ...
        ```

    Again, the ``*`` in the function signature enforces that all arguments after it have
    to be keyword-only arguments and this is a mandatory requirement for the decorator
    to work properly.

    """

    @wraps(apodization_function)
    def wrapped_apodization_function(
        x: RealNumericArrayLike,
        x_max: RealNumeric = 1.0,
        *,
        skip_validation: bool = False,
        **kwargs,
    ) -> NDArray[np.float64]:

        # --- Input Validation ---

        x_internal = get_validated_real_numeric_1d_array_like(
            value=x,
            name="x",
            min_size=1,
            output_dtype=np.float64,
        )

        if not skip_validation:
            x_max = get_validated_xmax(x_max=x_max)

        # --- Computation ---

        # the apodization function is only computed on the scaled interval [0, 1]
        x_internal = np.abs(x_internal) / x_max
        mask = x_internal <= 1.0

        apodization_values = np.empty_like(x_internal, dtype=np.float64)
        if mask.any():
            indices = np.where(mask)[0]
            apodization_values[indices] = apodization_function(
                x=x_internal[indices],
                x_max=x_max,
                skip_validation=skip_validation,
                **kwargs,
            )

        # the apodization function is zero outside the interval [-1, 1]
        mask = np.invert(mask)
        if mask.any():
            indices = np.where(mask)[0]
            apodization_values[indices] = 0.0

        return apodization_values

    return wrapped_apodization_function


def as_apodization_function(
    apodization_function: ApodizationFunction,
) -> WrappedApodizationFunction:
    """
    A decorator that checks if the given function is a valid apodization function and
    wraps it with the :func:`_convert_to_validated_apodization_function` decorator.

    Please refer to the Notes section for the requirements that the apodization function
    has to fulfill.

    Parameters
    ----------
    apodization_function : callable
        The function to check and wrap.
        It has to meet the requirements of an apodization function as described in the
        Notes section.

    Returns
    -------
    wrapped_apodization : callable
        The wrapped apodization function.

    Raises
    ------
    TypeError
        If the given function is not a function.
    ValueError
        If the given function is not a valid apodization function.

    Notes
    -----
    This decorator allows for wrapping the given apodization function which is only
    defined within the x-interval ``[0, 1]``, where ``0` corresponds to the center of
    the apodization and ``1`` to the right boundary where it fades out (not necessarily
    to zero).

    The wrapper

    - symmetrizes the apodization function to the interval ``[-1, 1]`` by exploiting the
        even symmetry of apodization functions, i.e., ``f(-x) = f(x)``,
    - scales the interval ``[-1, 1]`` to ``[-x_max, x_max]``,

    x-values outside the interval ``[-x_max, x_max]`` are handled naturally by not even
    evaluating the apodization function at those points.

    For this purpose, the apodization function is expected to have the following
    signature:

        ```python
        def apodization_function(
            x: RealNumericArrayLike,
            x_max: RealNumeric,
            *,  # <- mandatory enforcement of keyword-only arguments
            ...  # <- additional keyword arguments required by the apodization function
            skip_validation: bool = False,
        ) -> NDArray[np.float64]:
            ...
        ```

    So for example, the ``boxcar`` function would be defined as:

        ```python
        def boxcar(
            x: RealNumericArrayLike,
            x_max: RealNumeric = 1.0,
            *,
            skip_validation: bool = False,
        ) -> NDArray[np.float64]:
            ...
        ```

    while the ``zero_mapped_hyperbolic_sine`` that takes the additional argument
    ``exponent`` would be defined as:

        ```python
        def zero_mapped_hyperbolic_sine(
            x: RealNumericArrayLike,
            x_max: RealNumeric = 1.0,
            *,
            exponent: RealNumeric = 5.0,
            skip_validation: bool = False,
        ) -> NDArray[np.float64]:
            ...
        ```

    Again, the ``*`` in the function signature enforces that all arguments after it have
    to be keyword-only arguments and this is a mandatory requirement for the decorator
    to work properly.

    """

    if not inspect.isfunction(apodization_function):
        raise TypeError(
            f"The given apodization function '{apodization_function}' is "
            f"not a function."
        )

    # it is ensured that the function has the correct signature
    function_name = apodization_function.__name__
    signature = inspect.signature(apodization_function)
    required_parameters = [
        "x",
        "x_max",
        "skip_validation",
    ]

    for name, parameter in signature.parameters.items():
        # if the parameter is a required one, it is removed from the list of required
        # parameters to check if all required parameters are present afterwards
        if name in required_parameters:
            required_parameters.remove(name)

        # it is ensured that the function has ``x`` and ``x_max`` as the first and
        # second parameter, respectively
        if name in {"x", "x_max"}:
            if parameter.kind not in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.POSITIONAL_ONLY,
            }:
                raise ValueError(
                    f"The parameter '{name}' of the apodization function "
                    f"'{function_name}' has to be a positional or keyword argument.",
                )

            continue

        # for all other parameters, it is ensured that they are keyword-only arguments
        if parameter.kind != inspect.Parameter.KEYWORD_ONLY:
            raise ValueError(
                f"The parameter '{name}' of the apodization function '{function_name}' "
                f"has to be a keyword-only argument.",
            )

    # if any of the required parameters has not been found, an error is raised
    if len(required_parameters) > 0:
        raise ValueError(
            f"The apodization function '{function_name}' is missing the following "
            f'required parameters:\n{", ".join(required_parameters)}.',
        )

    wrapped_apodization_function = _convert_to_validated_apodization_function(
        apodization_function=apodization_function
    )

    # a tag is added to the wrapped function to indicate that it is a validated
    # apodization function
    wrapped_apodization_function.is_validated_apodization_function = True  # type: ignore

    return wrapped_apodization_function


@as_apodization_function
def not_implemented_apodization(
    x: RealNumericArrayLike,
    x_max: RealNumeric = 1.0,
    *,
    skip_validation: bool = False,
) -> NDArray[np.float64]:
    """
    A fake apodization function that raises a :class:`NotImplementedError` when called.

    """

    raise NotImplementedError("This apodization function is not yet implemented.")


@as_apodization_function
def boxcar(
    x: RealNumericArrayLike,
    x_max: RealNumeric = 1.0,
    *,
    skip_validation: bool = False,
) -> NDArray[np.float64]:
    """
    Computes the values of the boxcar function at the given points.

    Parameters
    ----------
    x : Array-like of shape (n,)
        The points at which to evaluate the apodization function.
        Negative entries are converted to positive ones under the assumption that
        the apodization function has even symmetry.
        Its length has to be at least 1.
        It is internally promoted to ``np.float64``.
    x_max : :class:`float` or :class:`int`, default=``1.0``
        The maximum value of the x-range over which the apodization function is defined.
        It must be a positive real number ``> 0``.
        With this, the x-range of ``[-1, 1]`` where apodization functions are typically
        defined is scaled to ``[-x_max, x_max]``.
    skip_validation : :class:`bool`, default=``False`` (keyword-only)
        Whether to skip the input validation of ``x_max`` (``True``) or not
        (``False``).
        ``x`` is always validated.
        This variable is meant for internal use only and it is highly discouraged to
        set it to ``True``.

    Returns
    -------
    apodization_values : :class:`numpy.ndarray` of shape (n,) of dtype ``np.float64``
        The values of the boxcar function at the given points.

    Notes
    -----
    The boxcar function is simply 1 for all points within the interval
    ``[-x_max, x_max]`` and 0 otherwise.

    """

    return np.ones_like(x, dtype=np.float64)


@as_apodization_function
def triangular(
    x: RealNumericArrayLike,
    x_max: RealNumeric = 1.0,
    *,
    skip_validation: bool = False,
) -> NDArray[np.float64]:
    """
    Computes the values of the triangular function at the given points.

    Parameters
    ----------
    x : Array-like of shape (n,)
        The points at which to evaluate the apodization function.
        Negative entries are converted to positive ones under the assumption that
        the apodization function has even symmetry.
        Its length has to be at least 1.
        It is internally promoted to ``np.float64``.
    x_max : :class:`float` or :class:`int`, default=``1.0``
        The maximum value of the x-range over which the apodization function is defined.
        It must be a positive real number ``> 0``.
        With this, the x-range of ``[-1, 1]`` where apodization functions are typically
        defined is scaled to ``[-x_max, x_max]``.
    skip_validation : :class:`bool`, default=``False`` (keyword-only)
        Whether to skip the input validation of ``x_max`` (``True``) or not
        (``False``).
        ``x`` is always validated.
        This variable is meant for internal use only and it is highly discouraged to
        set it to ``True``.

    Returns
    -------
    apodization_values : :class:`numpy.ndarray` of shape (n,) of dtype ``np.float64``
        The values of the triangular function at the given points.

    Notes
    -----
    The triangular function is a linear function that ramps up from 0 to 1 within the
    interval ``[-x_max, 0]`` and then ramps down again from 1 to 0 within the interval
    ``[0, x_max]``.

    """

    return 1.0 - x  # type: ignore


@as_apodization_function
def zero_mapped_hyperbolic_sine(
    x: RealNumericArrayLike,
    x_max: RealNumeric = 1.0,
    *,
    exponent: RealNumeric = 5.0,
    skip_validation: bool = False,
) -> NDArray[np.float64]:
    """
    Computes the values of the zero-mapped hyperbolic sine function at the given points.

    Parameters
    ----------
    x : Array-like of shape (n,)
        The points at which to evaluate the apodization function.
        Negative entries are converted to positive ones under the assumption that
        the apodization function has even symmetry.
        Its length has to be at least 1.
        It is internally promoted to ``np.float64``.
    x_max : :class:`float` or :class:`int`, default=``1.0``
        The maximum value of the x-range over which the apodization function is defined.
        It must be a positive real number ``> 0``.
        With this, the x-range of ``[-1, 1]`` where apodization functions are typically
        defined is scaled to ``[-x_max, x_max]``.
    exponent : :class:`float` or :class:`int`, default=``5.0`` (keyword-only)
        The exponent to which the apodization function is raised.
        It must be a real number ``>= 0``.
        ``0`` will result in a :func:`boxcar` function.
    skip_validation : :class:`bool`, default=``False`` (keyword-only)
        Whether to skip the input validation of ``x_max`` and ``exponent`` (``True``) or
        not (``False``).
        ``x`` is always validated.
        This variable is meant for internal use only and it is highly discouraged to
        set it to ``True``.

    Returns
    -------
    apodization_values : :class:`numpy.ndarray` of shape (n,) of dtype ``np.float64``
        The values of the zero-mapped hyperbolic sine function at the given points.

    Notes
    -----
    The zero-mapped hyperbolic sine function is defined as

    ``f(x) = (sinh(1 - (x / x_max)**2))**exponent / (sinh(1))**exponent``

    within the interval ``[-x_max, x_max]``. At the boundaries, the function fades out
    to zero in a second order continuous manner (i.e., the function and its first
    two derivatives are all continuously fading out to zero).

    """

    if not skip_validation:
        exponent = get_validated_real_numeric(
            value=exponent,
            name="exponent",
            min_value=0.0,
            min_inclusive=True,
        )

    return np.power(
        (np.sinh(1.0 - x * x)) / np.sinh(1.0),  # type: ignore
        exponent,
    )
