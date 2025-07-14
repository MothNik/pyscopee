"""
Mod :mod:`_validate._arrays`

This module provides input validations for array-like data structures to be used across
the ``pyscopee`` package.

"""

# === Setup ===

__all__ = [
    "get_validated_numeric_nd_array_like",
    "get_validated_real_numeric_1d_array_like",
    "get_validated_real_numeric_2d_array_like",
    "validate_1d_array_is_sorted",
    "validate_1d_array_is_evenly_spaced",
]

# === Imports ===

import operator
from enum import Enum
from typing import Any, List, Literal, Optional, Tuple, Type, Union, overload

import numpy as np

# === Models ===


class NumPyDTypeKinds(str, Enum):
    """
    The kinds of NumPy data types for data type checking.

    """

    NUMERIC = "biufc"  # boolean, integer, unsigned integer, float, complex
    NUMERIC_NO_BOOL = "iufc"
    REAL_NUMERIC = "biuf"
    REAL_NUMERIC_NO_BOOL = "iuf"
    INTEGER = "iu"
    FLOAT_OR_COMPLEX = "fc"

    def to_error_message_str(self) -> str:
        return {
            NumPyDTypeKinds.NUMERIC_NO_BOOL: "numeric (boolean excluded)",
            NumPyDTypeKinds.REAL_NUMERIC_NO_BOOL: "real numeric (boolean excluded)",
        }.get(self, self.name.lower().replace("_", " "))


# === Functions ===


@overload
def get_validated_numeric_nd_array_like(
    value: Any,
    name: str,
    dim: Literal[1],
    shape_limits: List[Tuple[Optional[int], Optional[int]]],
    dtype_kind: NumPyDTypeKinds,
    enforce_finite: bool,
    output_dtype: Optional[Type],
) -> np.ndarray[Tuple[int], np.dtype]: ...


@overload
def get_validated_numeric_nd_array_like(
    value: Any,
    name: str,
    dim: Literal[2],
    shape_limits: List[Tuple[Optional[int], Optional[int]]],
    dtype_kind: NumPyDTypeKinds,
    enforce_finite: bool,
    output_dtype: Optional[Type],
) -> np.ndarray[Tuple[int, int], np.dtype]: ...


def get_validated_numeric_nd_array_like(
    value: Any,
    name: str,
    dim: Literal[1, 2],
    shape_limits: List[Tuple[Optional[int], Optional[int]]],
    dtype_kind: NumPyDTypeKinds,
    enforce_finite: bool,
    output_dtype: Optional[Type],
) -> Union[np.ndarray[Tuple[int], np.dtype], np.ndarray[Tuple[int, ...], np.dtype]]:
    """
    Checks if a value is an N-dimensional Array-like of correct numeric data type and
    returns it as a NumPy N-dimensional Array.

    Parameters
    ----------
    value: any
        The value to check.
    name : :obj:`str`
        The name of the value used for error messages.
    dim : {``1``, ``2``}
        The expected dimensionality of the Array-like.
    shape_limits : [(:obj:`int` or ``None``, :obj:`int` or ``None``), ...]
        The expected shape limits of the Array-like.
        Its ``i``-th element is a tuple of the minimum and maximum allowed size of the
        ``i``-th dimension.
        If a limit is ``None``, the size is not checked against the respective bound.
        The length of the list must be equal to ``dim``.
    dtype_kind : :obj:`NumPyDTypeKinds`
        The kind of NumPy data types for data type checking.
    enforce_finite : :obj:`bool`
        Whether to enforce that all entries are finite, i.e., not ``nan``, ``-inf``, or
        ``inf`` (``True``) or not (``False``).
    output_dtype : :obj:`type` or ``None``, default=``None``
        The data type of the output NumPy Array.
        If ``None``, the data type is not changed.
        The conversion is done with ``value.astype(output_dtype, casting="safe")``.

    Returns
    -------
    checked_value : :obj:`numpy.ndarray` of shape (n, ...)
        The checked value.

    Raises
    ------
    ValueError
        If ``value`` is not or cannot be converted to a N-dimensional NumPy Array.
    ValueError
        If ``value`` is an empty Array.
    ValueError
        If ``value`` is not a N-dimensional Array-like.
    ValueError
        If ``shape_limits`` contains invalid limits.
    TypeError
        If ``value`` does not have the correct data type.
    AssertionError
        (Internal) If ``shape_limits`` is not of length ``dim``.

    """

    # first, the value is converted to a NumPy Array
    # NOTE: the case of the value being a NumPy Array is handled first to avoid
    #       unnecessary overhead
    if isinstance(value, np.ndarray):
        value_array = value
    else:

        array_converter = {
            1: np.atleast_1d,
            2: np.atleast_2d,
        }[dim]

        try:
            value_array = array_converter(value)
        except Exception as err:
            raise ValueError(
                f"'{name}' could not be converted to a NumPy Array-like."
            ) from err

    # empty Arrays are considered invalid
    if value_array.size < 1:
        raise ValueError(f"Expected '{name}' to be a non-empty Array-like.")

    # then, the value is checked to be a N-dimensional Array
    if value_array.ndim != dim:
        raise ValueError(
            f"Expected '{name}' to be a {dim}D Array-like, but got a "
            f"{value_array.ndim}D Array-like of shape {value_array.shape}."
        )

    # if the shape limits are not of length dim, an error is raised
    if len(shape_limits) != dim:
        raise AssertionError(
            f"Expected 'shape_limits' to be of length {dim}, but got a length of "
            f"{len(shape_limits)}."
        )

    # if a size is provided, the value is checked to have the expected size
    for axis, (min_size, max_size) in enumerate(shape_limits):
        axis_size = value_array.shape[axis]

        for size_bound, comparison in [
            (min_size, operator.ge),
            (max_size, operator.le),
        ]:
            if size_bound is None:
                continue

            if not comparison(axis_size, size_bound):
                raise ValueError(
                    f"Expected '{name}' to have a size between {min_size} and "
                    f"{max_size} for axis {axis}, but got a size of {axis_size}."
                )

    # afterwards, the value is checked for the correct data type
    if value_array.dtype.kind not in dtype_kind.value:
        raise TypeError(
            f"Expected '{name}' to be a {dim}D Array-like of "
            f"{dtype_kind.to_error_message_str()} values, but got a {dim}D Array-like "
            f"not meeting this requirement."
        )

    # if finite values are enforced, the value is checked to contain only finite values
    # NOTE: Python ``and`` evaluates the second argument lazily, i.e., if the first
    #       argument is ``False``, the second argument is not even evaluated
    if enforce_finite and not np.isfinite(value_array).all():
        raise TypeError(
            f"Expected '{name}' to contain only finite values, but got an Array with "
            f"non-finite values."
        )

    # if a new data type is provided, the value is converted to this data type
    if output_dtype is not None:
        if output_dtype != value_array.dtype:
            try:
                value_array = value_array.astype(output_dtype, casting="safe")
            except Exception as err:
                raise TypeError(
                    f"Could not convert '{name}' from a '{value_array.dtype}'- to a "
                    f"'{output_dtype.__name__}'-Array (uses 'safe' casting)."
                ) from err

    return value_array


def get_validated_real_numeric_1d_array_like(
    value: Any,
    name: str,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    enforce_finite: bool = False,
    output_dtype: Optional[Type] = None,
) -> np.ndarray[Tuple[int], np.dtype]:
    """
    Checks if a value is a 1D Array-like of real numeric values and returns it as a
    NumPy 1D Array.
    Booleans entries are excluded from the allowed data types.

    Parameters
    ----------
    value: any
        The value to check.
    name : :obj:`str`
        The name of the value used for error messages.
    min_size, max_size : :obj:`int` or ``None``, default=``None``
        The minimum and maximum allowed size of the 1D Array-like.
        If ``None``, the size is not checked against the respective bound.
        Arrays of size 0 will always be considered invalid.
    enforce_finite : :obj:`bool`, default=``False``
        Whether to enforce that all entries are finite, i.e., not ``nan``, ``-inf``, or
        ``inf`` (``True``) or not (``False``).
    output_dtype : :obj:`type` or ``None``, default=``None``
        The data type of the output NumPy Array.
        If ``None``, the data type is not changed.
        The conversion is done with ``value.astype(output_dtype, casting="safe")``.

    Returns
    -------
    checked_value : :obj:`numpy.ndarray` of shape (n, )
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

    return get_validated_numeric_nd_array_like(
        value=value,
        name=name,
        dim=1,
        shape_limits=[
            (min_size, max_size),
        ],
        dtype_kind=NumPyDTypeKinds.REAL_NUMERIC_NO_BOOL,
        enforce_finite=enforce_finite,
        output_dtype=output_dtype,
    )


def get_validated_real_numeric_2d_array_like(
    value: Any,
    name: str,
    rows_min_num: Optional[int] = None,
    rows_max_num: Optional[int] = None,
    columns_min_num: Optional[int] = None,
    columns_max_num: Optional[int] = None,
    enforce_finite: bool = False,
    output_dtype: Optional[Type] = None,
) -> np.ndarray:
    """
    Checks if a value is a 2D Array-like of real numeric values and returns it as a
    NumPy 2D Array.
    Booleans entries are excluded from the allowed data types.

    Parameters
    ----------
    value: any
        The value to check.
    name : :obj:`str`
        The name of the value used for error messages.
    rows_min_num, rows_max_num : :obj:`int` or ``None``, default=``None``
        The minimum and maximum allowed number of rows of the 2D Array-like.
        If ``None``, the number of rows is not checked against the respective bound.
    columns_min_num, columns_max_num : :obj:`int` or ``None``, default=``None``
        Equivalent to ``rows_min_num`` and ``rows_max_num`` but for the columns.
    enforce_finite : :obj:`bool`, default=``False``
        Whether to enforce that all entries are finite, i.e., not ``nan``, ``-inf``, or
        ``inf`` (``True``) or not (``False``).
    output_dtype : :obj:`type` or ``None``, default=``None``
        The data type of the output NumPy Array.
        If ``None``, the data type is not changed.
        The conversion is done with ``value.astype(output_dtype, casting="safe")``.

    Returns
    -------
    checked_value : :obj:`numpy.ndarray` of shape (n, m)
        The checked value.

    Raises
    ------
    ValueError
        If ``value`` is not or cannot be converted to a 2D NumPy Array.
    ValueError
        If ``value`` is an empty Array.
    ValueError
        If ``value`` is not a 2D Array-like.
    ValueError
        If ``rows_min_num <= value.shape[0] <= rows_max_num`` is not fulfilled.
    ValueError
        If ``columns_min_num <= value.shape[1] <= columns_max_num`` is not fulfilled.
    TypeError
        If ``value`` does not contain only real numeric values.

    """

    return get_validated_numeric_nd_array_like(
        value=value,
        name=name,
        dim=2,
        shape_limits=[
            (rows_min_num, rows_max_num),
            (columns_min_num, columns_max_num),
        ],
        dtype_kind=NumPyDTypeKinds.REAL_NUMERIC_NO_BOOL,
        enforce_finite=enforce_finite,
        output_dtype=output_dtype,
    )


def validate_1d_array_is_sorted(
    value: np.ndarray,
    name: str,
    order: Literal["ascending", "descending", "both"] = "ascending",
    strict: bool = True,
) -> None:
    """
    Validates that a 1D Array is sorted in either (strictly) ascending and/or (strictly)
    descending order.

    Parameters
    ----------
    value : :obj:`numpy.ndarray` of shape (n, )
        The 1D Array to validate.
        It may not be empty.
    name : :obj:`str`
        The name of the value used for error messages.
    order : {``"ascending"``, ``"descending"``, ``"both"``}, default=``"ascending"``
        The order to check.
    strict : :obj:`bool`, default=``True``
        Whether to check for strict ordering (``True``) or not (``False``), i.e., if
        subsequent values are allowed to be equal or not.

    Raises
    ------
    TypeError
        If ``value`` is not a NumPy Array.
    ValueError
        If ``value`` is not a 1D Array.
    ValueError
        If ``value`` is empty.
    ValueError
        If ``order`` is not one of the supported options.
    ValueError
        If ``value`` is not sorted in the specified order.

    """

    # first, the value is checked to be a 1D Array
    if not isinstance(value, np.ndarray):
        raise TypeError(
            f"Expected '{name}' to be a NumPy Array, but got {type(value)}."
        )

    if value.ndim != 1:
        raise ValueError(
            f"Expected '{name}' to be a 1D Array, but got a {value.ndim}D Array."
        )

    # empty Arrays are considered invalid
    if value.size < 1:
        raise ValueError(f"Expected '{name}' to be a non-empty Array.")

    # then, the value is checked to be sorted in the specified order
    order = order.lower()  # type: ignore
    if order not in {"ascending", "descending", "both"}:
        raise ValueError(
            f"Expected 'order' to be one of 'ascending', 'descending', or 'both', but "
            f"got '{order}'."
        )

    if order in {"ascending", "both"}:
        comparison_operator = operator.lt if strict else operator.le
        if (comparison_operator(value[0:-1], value[1:])).all():
            return

    if order in {"descending", "both"}:
        comparison_operator = operator.gt if strict else operator.ge
        if (comparison_operator(value[0:-1], value[1:])).all():
            return

    strict_str = "strict" if strict else ""
    order_str = order if order != "both" else "ascending or descending"

    raise ValueError(
        f"Expected '{name}' to be sorted in {strict_str} {order_str} order, but got an "
        f"Array that is not sorted in this way."
    )


def validate_1d_array_is_evenly_spaced(
    value: np.ndarray,
    name: str,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> None:
    """
    Validates that a 1D Array is evenly spaced in either ascending or descending order.

    Parameters
    ----------
    value : :obj:`numpy.ndarray` of shape (n, )
        The 1D Array to validate.
        It may not be empty.
        It is promoted to ``numpy.float64`` for the comparison.
    name : :obj:`str`
        The name of the value used for error messages.
    atol, rtol : :obj:`float`, default=``1e-8`` and ``1e-5``
        The absolute and relative tolerances for the spacing checks that will be
        passed to :func:`numpy.allclose` as ``np.allclose(value, reference, atol=atol, rtol=rtol)``.
        ``reference`` is created by :func:`numpy.linspace` as
        ``np.linspace(value[0], value[-1], num=value.size)``.

    Raises
    ------
    TypeError
        If ``value`` is not a NumPy Array.
    ValueError
        If ``value`` is not a 1D Array.
    TypeError
        If the entries of ``value`` cannot be promoted to ``numpy.float64``.
    ValueError
        If ``value`` is empty.
    ValueError
        If ``value`` is not evenly spaced in either ascending or descending order.

    """  # noqa: E501

    # first, the value is checked to be a 1D Array
    if not isinstance(value, np.ndarray):
        raise TypeError(
            f"Expected '{name}' to be a NumPy Array, but got {type(value)}."
        )

    if value.ndim != 1:
        raise ValueError(
            f"Expected '{name}' to be a 1D Array, but got a {value.ndim}D Array."
        )

    # empty Arrays are considered invalid
    if value.size < 1:
        raise ValueError(f"Expected '{name}' to be a non-empty Array.")

    # then, the value is checked to be evenly spaced in either ascending or descending
    # order
    # this is done by creating a reference ``linspace`` and checking if the values are
    # numerically close to it
    # NOTE: to ensure numerical stability, the values are promoted to ``numpy.float64``
    #       if they are not already
    if value.dtype != np.float64:
        try:
            value = value.astype(np.float64)
        except Exception as error:
            raise TypeError(
                f"Could not convert '{name}' to a NumPy Array of dtype 'float64' for "
                f"checking if it is evenly spaced."
            ) from error

    # NOTE: this naturally handles ascending and descending order simultaneously
    reference = np.linspace(
        start=value[0],
        stop=value[-1],
        num=value.size,
        dtype=np.float64,
    )

    # if the values are numerically close to evenly spaced values, everything is fine
    if np.allclose(
        value,
        reference,
        atol=atol,
        rtol=rtol,
    ):
        return

    # otherwise, if the values are not numerically close to evenly spaced values, an
    # error is raised
    raise ValueError(
        f"Expected '{name}' to be evenly spaced in either ascending or descending "
        f"order, but got an Array with uneven spacing."
    )
