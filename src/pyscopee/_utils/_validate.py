"""
Mod :mod:`_utils._validate`

This module provides input validations used across the ``pyscopee`` package.

"""

# === Setup ===

__all__ = [
    "get_validated_integer",
    "get_validated_real_numeric",
    "get_validated_real_numeric_1d_array_like",
]

# === Imports ===

import operator
from enum import IntEnum
from typing import Any, Optional, Tuple, Type, TypeVar

import numpy as np

# === Types ===

ValueType = TypeVar("ValueType", int, float)

# === Models ===


class _BoundKind(IntEnum):
    """
    The kinds of bounds for comparisons.

    """

    LOW = 0
    UPPER = 1


# === Auxiliary Functions ===


def _convert_to_validated_type(
    value: Any,
    name: str,
    output_type: Type[ValueType],
    allowed_from_types: Tuple[Type, ...],
) -> ValueType:
    """
    Converts a value to a specific type and checks if it is one of the allowed types.

    Parameters
    ----------
    value : Any
        The value to convert.
    name : :class:`str`
        The name of the value used for error messages.
    output_type : type
        The type to convert the value to.
        It should not be part of ``allowed_from_types``.
    allowed_from_types : (type, ...)
        The allowed types for the value from which it can be converted.
        It should not contain ``output_type``.

    Returns
    -------
    converted_value : output_type
        The converted value.

    Raises
    ------
    TypeError
        If ``value``'s type is neither ``output_type`` nor one of
        ``allowed_from_types``.

    """

    # if the value is already of the output type, it is returned without conversion
    if isinstance(value, output_type):
        return value

    # otherwise, if it is one of the allowed types, it is converted to the output type
    if isinstance(value, allowed_from_types):
        return output_type(value)

    # if the value is neither of the output type nor one of the allowed types, an error
    # is raised
    # NOTE: the following slices [7:-1] # removes the "<class '" and "'>" parts
    allowed_types_names = f"{output_type}"[7:-1]
    allowed_types_names += " / "
    allowed_types_names += " / ".join([f"{atp}"[7:-1] for atp in allowed_from_types])
    value_type = f"{type(value)}"[7:-1]
    raise TypeError(
        f"Expected '{name}' to be of type {allowed_types_names}, but got {value_type}."
    )


def _get_bound_validated_value(
    value: ValueType,
    name: str,
    bound: Optional[ValueType] = None,
    bound_kind: _BoundKind = _BoundKind.LOW,
    clip: bool = False,
) -> ValueType:
    """
    Checks if a value satisfies a comparison with a bound, clips it if necessary, and
    returns it.

    Parameters
    ----------
    value : :class:`float` or :class:`int`
        The value to check.
    name : :class:`str`
        The name of the value used for error messages.
    bound : :class:`float` or :class:`int` or ``None``, default=``None``
        The bound to compare against.
        If ``None``, no comparison is performed.
    bound_kind : :class:`_BoundKind`, default=``_BoundKind.LOW``
        The bound kind to compare against, i.e., either the lower or upper bound.
    clip : :class:`bool`, default=``False``
        Whether to clip the ``value`` to the `bound`` if the comparison is not
        satisfied.
        For

        - ``bound_kind=_BoundKind.LOW``, the conversion is
            ``value = max(value, bound)``
        - ``bound_kind=_BoundKind.UPPER``, the conversion is
            ``value = min(value, bound)``.

    Returns
    -------
    validated_value : :class:`float` or :class:`int`
        The validated value.

    Raises
    ------
    ValueError
        If the comparison is not satisfied and ``clip`` is ``False``.

    """

    # if no bound is provided, the function returns without doing anything
    if bound is None:
        return value

    # the comparison operator, clip function, and message string are determined
    if bound_kind == _BoundKind.LOW:
        comparison_operator = operator.ge
        comparison_str = ">="
        clipper = max
    else:
        comparison_operator = operator.le
        comparison_str = "<="
        clipper = min

    # the comparison is performed and an error is raised if it is not satisfied
    comparison_satisfied = comparison_operator(value, bound)
    if not comparison_satisfied and not clip:
        raise ValueError(
            f"Expected '{name}' to be {comparison_str} {bound}, but got {value}."
        )

    if not comparison_satisfied and clip:
        value = clipper(value, bound)

    return value


def _get_validated_scalar(
    value: Any,
    name: str,
    output_type: Type[ValueType],
    allowed_from_types: Tuple[Type, ...],
    min_value: Optional[ValueType] = None,
    max_value: Optional[ValueType] = None,
    clip: bool = False,
) -> ValueType:
    """
    Validates a scalar value by converting it to a specific type and checking if it is
    within a specific range.

    Parameters
    ----------
    value : Any
        The value to validate.
    name : :class:`str`
        The name of the value used for error messages.
    output_type : type
        The type to convert the value to.
        It should not be part of ``allowed_from_types``.
    allowed_from_types : (type, ...)
        The allowed types for the value from which it can be converted.
        It should not contain ``output_type``.
    min_value, max_value : :class:`float` or :class:`int` or ``None``, default=``None``
        The minimum and maximum allowed values.
        If ``None``, the value is not checked against the respective bound.
    clip : :class:`bool`, default=``False``
        Whether to clip the value to the allowed range if it is not within
        [``min_value``, ``max_value``].

    Returns
    -------
    validated_value : :class:`float` or :class:`int`
        The validated value.

    Raises
    ------
    TypeError
        If ``value`` is not one of the allowed types.
    ValueError
        If ``value`` is not within the allowed range and ``clip`` is ``False``.
    AssertionError
        If ``min_value`` is greater than ``max_value`` (if both are not ``None``).

    """

    # first, the value is converted to the output type and checked to be one of the
    # allowed types
    value = _convert_to_validated_type(
        value=value,
        name=name,
        output_type=output_type,
        allowed_from_types=allowed_from_types,
    )

    # if both bounds are provided, the minimum bound is checked to be less than the
    # maximum bound
    if min_value is not None and max_value is not None:
        if min_value > max_value:
            raise ValueError(
                f"Expected minimum value for '{name}' to be <= maximum value, but got "
                f"min = {min_value} and max = {max_value}."
            )

    # afterwards, the value is checked to be within the allowed range and clipped if
    # necessary and enabled
    for bound, bound_kind in [
        (min_value, _BoundKind.LOW),
        (max_value, _BoundKind.UPPER),
    ]:
        value = _get_bound_validated_value(
            value=value,
            name=name,
            bound=bound,
            bound_kind=bound_kind,  # type: ignore
            clip=clip,
        )

    return value


# === Functions ===


def get_validated_integer(
    value: Any,
    name: str,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    clip: bool = False,
) -> int:
    """
    Checks if a value is an integer and returns it as an integer.

    Parameters
    ----------
    value : Any
        The value to check.
    name : :class:`str`
        The name of the value used for error messages.
    min_value, max_value : :class:`int` or ``None``, default=``None``
        The minimum and maximum allowed values.
        If ``None``, the value is not checked against the respective bound.
    clip : :class:`bool`, default=``False``
        Whether to clip the value to the allowed range if it is not within
        [``min_value``, ``max_value``].

    Returns
    -------
    checked_value : :class:`int`
        The checked value as an integer.

    Raises
    ------
    TypeError
        If ``value`` is not an integer type.
    ValueError
        If ``value`` is not within the allowed range and ``clip`` is ``False``.

    """

    return _get_validated_scalar(
        value=value,
        name=name,
        output_type=int,
        allowed_from_types=(np.integer,),
        min_value=min_value,
        max_value=max_value,
        clip=clip,
    )


def get_validated_real_numeric(
    value: Any,
    name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    clip: bool = False,
) -> float:
    """
    Checks if a value is a real numeric and returns it as a float.

    Parameters
    ----------
    value : Any
        The value to check.
    name : :class:`str`
        The name of the value used for error messages.
    min_value, max_value : :class:`float` or ``None``, default=``None``
        The minimum and maximum allowed values.
        If ``None``, the value is not checked against the respective bound.
    clip : :class:`bool`, default=``False``
        Whether to clip the value to the allowed range if it is not within
        [``min_value``, ``max_value``].

    Returns
    -------
    checked_value : :class:`float`
        The checked value as a float.

    Raises
    ------
    TypeError
        If ``value`` is not real numeric.
    ValueError
        If ``value`` is not within the allowed range and ``clip`` is ``False``.

    """

    return _get_validated_scalar(
        value=value,
        name=name,
        output_type=float,
        allowed_from_types=(
            np.floating,
            int,
            np.integer,
        ),
        min_value=min_value,
        max_value=max_value,
        clip=clip,
    )


def get_validated_real_numeric_1d_array_like(
    value: Any,
    name: str,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    output_dtype: Optional[Type] = None,
) -> np.ndarray:
    """
    Checks if a value is a 1D Array-like of real numeric values and returns it as a
    NumPy 1D Array.

    Parameters
    ----------
    value : Any
        The value to check.
    name : :class:`str`
        The name of the value used for error messages.
    min_size, max_size : :class:`int` or ``None``, default=``None``
        The minimum and maximum allowed size of the 1D Array-like.
        If ``None``, the size is not checked against the respective bound.
        Arrays of size 0 will always be considered invalid.
    output_dtype : :class:`type` or ``None``, default=``None``
        The data type of the output NumPy Array.
        If ``None``, the data type is not changed.

    Returns
    -------
    checked_value : :class:`numpy.ndarray` of shape (n, )
        The checked value.

    Raises
    ------
    ValueError
        If ``value`` is not or cannot be converted to a 1D NumPy Array.
    ValueError
        If ``value`` is an empty Array.
    ValueError
        If ``value`` is not a 1D Array-like.
    ValueError
        If ``min_size <= value.size <= max_size`` is not fulfilled.
    TypeError
        If ``value`` does not contain only real numeric values.

    """

    # first, the value is converted to a NumPy Array
    # NOTE: the case of the value being a NumPy Array is handled first to avoid
    #       unnecessary overhead
    if isinstance(value, np.ndarray):
        value_array = value
    else:
        try:
            value_array = np.atleast_1d(value)
        except Exception as err:
            raise ValueError(
                f"'{name}' could not be converted to a NumPy Array-like"
            ) from err

    # empty Arrays are considered invalid
    if value_array.size < 1:
        raise ValueError(f"Expected '{name}' to be a non-empty Array-like.")

    # then, the value is checked to be a 1D Array
    if value_array.ndim != 1:
        raise ValueError(
            f"Expected '{name}' to be a 1D Array-like, but got a "
            f"{value_array.ndim}D Array of shape {value_array.shape}."
        )

    # if a size is provided, the value is checked to have the expected size
    for size_bound, comparison in [(min_size, operator.ge), (max_size, operator.le)]:
        if size_bound is None:
            continue

        if not comparison(value_array.size, size_bound):
            raise ValueError(
                f"Expected '{name}' to have a size between {min_size} and {max_size}, "
                f"but got a size of {value_array.size}."
            )

    # afterwards, the value is checked to be a 1D Array of real numeric values
    if not np.isreal(value_array).all():
        raise TypeError(
            f"Expected '{name}' to be a 1D Array-like of real numeric values, but got "
            f"a 1D Array-like with non-real numeric values."
        )

    # if a new data type is provided, the value is converted to this data type
    if output_dtype is not None:
        if output_dtype != value_array.dtype:
            value_array = value_array.astype(output_dtype)

    return value_array
