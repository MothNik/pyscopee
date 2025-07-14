"""
Mod :mod:`_validate._numbers`

This module provides input validations for numeric data to be used across the
``pyscopee`` package.

"""

# === Setup ===

__all__ = [
    "get_validated_integer",
    "get_validated_real_numeric",
    "isinstance_incl_none",
]

# === Imports ===

import operator
from enum import IntEnum
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar, Union

import numpy as np

# === Types ===

_ValueType = TypeVar("_ValueType", int, float)

# === Models ===


class _BoundKind(IntEnum):
    """
    The kinds of bounds for comparisons.

    """

    LOW = 0
    UPPER = 1


# === Constants ===

# the assignment of the comparison operators for the bounds
_BOUND_COMPARISON_ASSIGNMENT: Dict[
    Tuple[_BoundKind, bool],
    Tuple[Callable[[Any, Any], bool], Callable[[Any, Any], Any], str],
] = {
    (_BoundKind.LOW, True): (operator.ge, max, ">="),
    (_BoundKind.LOW, False): (operator.gt, max, ">"),
    (_BoundKind.UPPER, True): (operator.le, min, "<="),
    (_BoundKind.UPPER, False): (operator.lt, min, "<"),
}

# === Auxiliary Functions ===


def _convert_to_validated_type(
    value: Any,
    name: str,
    output_type: Type[_ValueType],
    allowed_from_types: Tuple[Type, ...],
) -> _ValueType:
    """
    Converts a value to a specific type and checks if it is one of the allowed types.

    Parameters
    ----------
    value: any
        The value to convert.
    name : :obj:`str`
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
    value: _ValueType,
    name: str,
    bound: Optional[_ValueType],
    bound_kind: _BoundKind,
    bound_inclusive: bool,
    clip: bool,
) -> _ValueType:
    """
    Checks if a value satisfies a comparison with a bound, clips it if necessary, and
    returns it.

    Parameters
    ----------
    value : :obj:`float` or :obj:`int`
        The value to check.
    name : :obj:`str`
        The name of the value used for error messages.
    bound : :obj:`float` or :obj:`int` or ``None``
        The bound to compare against.
        If ``None``, no comparison is performed.
    bound_kind : :obj:`_BoundKind`
        The bound kind to compare against, i.e., either the lower or upper bound.
    bound_inclusive : :obj:`bool`
        Whether the bound comparison is inclusive (with comparison operators ``>=`` and
        ``<=``; ``True``) or exclusive (with comparison operators ``>`` and ``<``;
        ``False``).
    clip : :obj:`bool`
        Whether to clip the ``value`` to the `bound`` if the comparison is not
        satisfied.
        For

        - ``bound_kind=_BoundKind.LOW``, the conversion is
            ``value = max(value, bound)``
        - ``bound_kind=_BoundKind.UPPER``, the conversion is
            ``value = min(value, bound)``.

        It cannot be ``True`` if ``bound_inclusive`` is ``False``.

    Returns
    -------
    validated_value : :obj:`float` or :obj:`int`
        The validated value.

    Raises
    ------
    ValueError
        If the comparison is not satisfied and ``clip`` is ``False``.
    ValueError
        If ``clip`` is ``True`` and ``bound_inclusive`` is ``False``.

    """

    # if clipping is enabled and the bound is exclusive, an error is raised
    if clip and not bound_inclusive:
        raise ValueError(f"Cannot clip '{name}' to an exclusive bound. ")

    # if no bound is provided, the function returns without doing anything
    if bound is None:
        return value

    # the comparison operator, clip function, and message string are determined
    (
        comparison_operator,
        clipper,
        comparison_str,
    ) = _BOUND_COMPARISON_ASSIGNMENT[(bound_kind, bound_inclusive)]

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
    output_type: Type[_ValueType],
    allowed_from_types: Tuple[Type, ...],
    min_value: Optional[_ValueType],
    min_inclusive: bool,
    max_value: Optional[_ValueType],
    max_inclusive: bool,
    clip: bool,
) -> _ValueType:
    """
    Validates a scalar value by converting it to a specific type and checking if it is
    within a specific range.

    Parameters
    ----------
    value: any
        The value to validate.
    name : :obj:`str`
        The name of the value used for error messages.
    output_type : type
        The type to convert the value to.
        It should not be part of ``allowed_from_types``.
    allowed_from_types : (type, ...)
        The allowed types for the value from which it can be converted.
        It should not contain ``output_type``.
    min_value, max_value : :obj:`float` or :obj:`int` or ``None``
        The minimum and maximum allowed values.
        If ``None``, the value is not checked against the respective bound.
    min_inclusive, max_inclusive : :obj:`bool`
        Whether the minimum and maximum value bounds are inclusive (with comparison
        operators ``>=`` and ``<=``; ``True``) or exclusive (with comparison operators
        ``>`` and ``<``; ``False``).
    clip : :obj:`bool`
        Whether to clip the value to the allowed range if it is not within
        [``min_value``, ``max_value``].

    Returns
    -------
    validated_value : :obj:`float` or :obj:`int`
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
    for bound, bound_kind, is_inclusive in [
        (min_value, _BoundKind.LOW, min_inclusive),
        (max_value, _BoundKind.UPPER, max_inclusive),
    ]:
        value = _get_bound_validated_value(
            value=value,
            name=name,
            bound=bound,
            bound_kind=bound_kind,
            bound_inclusive=is_inclusive,
            clip=clip,
        )

    return value


# === Functions ===


def isinstance_incl_none(
    value: Any,
    types: Union[Type, Tuple[Optional[Type], ...], None],
) -> bool:
    """
    Checks if a value is an instance of one of the provided types including a check
    for ``None``.

    Parameters
    ----------
    value: any
        The value to check.
    types : type or (type, ...) or ``None``
        The types to check against.
        If a type is ``None``, a check is performed for ``None``.

    Returns
    -------
    is_instance : :obj:`bool`
        Whether the value is an instance of one of the provided types.

    """

    if not isinstance(types, tuple):
        types = (types,)

    if None in types:
        if value is None:
            return True

        # filter out None from the types to check against
        types = tuple(filter(lambda t: t is not None, types))

    return isinstance(value, types)  # type: ignore


def get_validated_integer(
    value: Any,
    name: str,
    min_value: Optional[int] = None,
    min_inclusive: bool = True,
    max_value: Optional[int] = None,
    max_inclusive: bool = True,
    clip: bool = False,
) -> int:
    """
    Checks if a value is an integer and returns it as an integer.

    Parameters
    ----------
    value : any
        The value to check.
    name : :obj:`str`
        The name of the value used for error messages.
    min_value, max_value : :obj:`int` or ``None``, default=``None``
        The minimum and maximum allowed values.
        If ``None``, the value is not checked against the respective bound.
    min_inclusive, max_inclusive : :obj:`bool`, default=``True``
        Whether the minimum and maximum value bounds are inclusive (with comparison
        operators ``>=`` and ``<=``; ``True``) or exclusive (with comparison operators
        ``>`` and ``<``; ``False``).
    clip : :obj:`bool`, default=``False``
        Whether to clip the value to the allowed range if it is not within
        [``min_value``, ``max_value``].

    Returns
    -------
    checked_value : :obj:`int`
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
        min_inclusive=min_inclusive,
        max_value=max_value,
        max_inclusive=max_inclusive,
        clip=clip,
    )


def get_validated_real_numeric(
    value: Any,
    name: str,
    min_value: Optional[float] = None,
    min_inclusive: bool = True,
    max_value: Optional[float] = None,
    max_inclusive: bool = True,
    clip: bool = False,
) -> float:
    """
    Checks if a value is a real numeric and returns it as a float.

    Parameters
    ----------
    value: any
        The value to check.
    name : :obj:`str`
        The name of the value used for error messages.
    min_value, max_value : :obj:`float` or ``None``, default=``None``
        The minimum and maximum allowed values.
        If ``None``, the value is not checked against the respective bound.
    min_inclusive, max_inclusive : :obj:`bool`, default=``True``
        Whether the minimum and maximum value bounds are inclusive (with comparison
        operators ``>=`` and ``<=``; ``True``) or exclusive (with comparison operators
        ``>`` and ``<``; ``False``).
    clip : :obj:`bool`, default=``False``
        Whether to clip the value to the allowed range if it is not within
        [``min_value``, ``max_value``].

    Returns
    -------
    checked_value : :obj:`float`
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
        min_inclusive=min_inclusive,
        max_value=max_value,
        max_inclusive=max_inclusive,
        clip=clip,
    )
