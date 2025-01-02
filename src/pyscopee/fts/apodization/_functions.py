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
    "print_apodization_function_template",
    "triangular",
    "zero_mapped_hyperbolic_sine",
]

# === Imports ===


import ast
import inspect
import textwrap
from functools import wraps
from typing import Callable, Optional, Protocol, Union

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


def _validate_apodization_function_signature(function: Callable) -> None:
    """
    Validates the signature of an apodization function.

    Parameters
    ----------
    function : callable
        The function to validate.

    Raises
    ------
    TypeError
        If the given function is not a function.
    ValueError
        If the given function does not have the correct signature in terms of the
        positional and keyword arguments.
    ValueError
        If the default value of ``x_max`` is not ``1.0``.

    """

    # --- Constants ---

    REFERENCE_SIGNATURE_MESSAGE = textwrap.dedent(
        """
        The apodization function is expected to have the following signature:

        ```python
        def {function_name}(
            x: RealNumericArrayLike,
            x_max: RealNumeric = 1.0,  # ← mandatory default value
            *,  # ← mandatory enforcement of keyword-only arguments
            ...  # ← additional keyword arguments required by the apodization function
            skip_validation: bool = False,
        ) -> NDArray[np.float64]:
            ...
        ```
        """
    )

    # --- Signature Validation ---

    if not inspect.isfunction(function):
        raise TypeError(
            f"The given apodization function '{function}' is not a function."
        )

    # it is ensured that the function has the correct signature
    function_name = function.__name__
    signature = inspect.signature(function)
    required_parameters = [
        "x",
        "x_max",
        "skip_validation",
    ]

    # a fast check is performed to see if all required parameters are present
    missing_parameters = set(required_parameters) - set(signature.parameters)
    if len(missing_parameters) > 0:
        missing_parameters = sorted([f"'{name}'" for name in missing_parameters])
        raise ValueError(
            f"The apodization function '{function_name}' is missing the following "
            f'required parameters:\n{", ".join(missing_parameters)}\n\n'
            + REFERENCE_SIGNATURE_MESSAGE.format(function_name=function_name),
        )

    # it needs to be ensured that ``x`` is the first positional or keyword argument
    # and ``x_max`` is the second one
    parameter_names = list(signature.parameters.keys())
    for position, reference_name in enumerate(("x", "x_max")):
        if parameter_names[position] != reference_name:
            raise ValueError(
                f"The apodization function '{function_name}' is expected to have "
                f"'{reference_name}' as the {position + 1}. parameter, but "
                f"'{parameter_names[position]}' is in this position.\n\n"
                + REFERENCE_SIGNATURE_MESSAGE.format(function_name=function_name),
            )

        parameter = signature.parameters[reference_name]
        if parameter.kind not in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        }:
            raise ValueError(
                f"The parameter '{reference_name}' of the apodization function "
                f"'{function_name}' has to be a positional or keyword argument.\n\n"
                + REFERENCE_SIGNATURE_MESSAGE.format(function_name=function_name),
            )

    # afterwards, all other parameters are checked for being keyword-only arguments
    for name, parameter in signature.parameters.items():
        if name in {"x", "x_max"}:
            continue

        # for all other parameters, it is ensured that they are keyword-only arguments
        if parameter.kind != inspect.Parameter.KEYWORD_ONLY:
            raise ValueError(
                f"The parameter '{name}' of the apodization function '{function_name}' "
                f"has to be a keyword-only argument.\n\n"
                + REFERENCE_SIGNATURE_MESSAGE.format(function_name=function_name),
            )

    # finally, it is ensured that the default value of ``x_max`` is ``1.0``
    x_max_default = signature.parameters["x_max"].default
    if x_max_default != 1.0:
        if x_max_default is not inspect.Parameter.empty:
            x_max_default_str = f"'{x_max_default}'"
        else:
            x_max_default_str = "not provided"

        raise ValueError(
            f"The default value of 'x_max' in the apodization function "
            f"'{function_name}' has to be '1.0', but it is {x_max_default_str}.\n\n"
            + REFERENCE_SIGNATURE_MESSAGE.format(function_name=function_name),
        )

    return


def _ensure_variable_is_not_used_in_function(
    function: Callable,
    variable_name: str,
) -> None:
    """
    Ensures that a variable is not used in the computation of a function.

    Parameters
    ----------
    function : callable
        The function to check.
    variable_name : :class:`str`
        The name of the variable to check.

    Raises
    ------
    ValueError
        If the variable is used in the computations of the function.

    """

    # the code is parsed to an AST to check if the variable is used
    function_source = inspect.getsource(function)
    function_ast = ast.parse(function_source)

    # every node in the AST is checked if it is a variable access and if it is the
    # variable that should not be used
    offending_line_number = None
    for node in ast.walk(function_ast):
        if isinstance(node, ast.Name) and node.id == variable_name:
            offending_line_number = node.lineno
            break

    # if the variable is not accessed, the function is valid
    if offending_line_number is None:
        return

    # NOTE: the following might not be very efficient, but it is only used for raising
    #       an error and not for the actual computation of the function
    # if the variable is accessed, the offending code lines are extracted
    function_source = function_source.split("\n")
    function_source = function_source[
        max(0, offending_line_number - 5) : min(
            offending_line_number + 5, len(function_source)
        )
    ]

    # for a detailed error message, the file name and absolute line number of the
    # offending code are determined
    function_file_name = inspect.getfile(function)
    offending_line_number_in_file = (
        function.__code__.co_firstlineno + offending_line_number - 1
    )

    # finally, the accesses to ``x_max`` are found in the offending code and emphasized
    # for an error message
    offending_code_lines = ""
    for line in function_source:
        # it is found if the variable name is used in a comment or actually valid code
        if variable_name not in line:
            offending_code_lines += f"{line}\n"
            continue

        # if a "#" is found before the variable name, it is a comment and the variable
        # is not used in the code
        if "#" in line:
            if line.index("#") < line.index(variable_name):
                offending_code_lines += f"{line}\n"
                continue

        # the variable name is emphasized in the error message (bold, underlined, and
        # with red color)
        line = line.replace(
            variable_name,
            f"\033[4m\033[1m\033[91m> > > {variable_name} < < <\033[0m",
        )

        offending_code_lines += f"{line}\n"

    raise ValueError(
        f"The variable '{variable_name}' is not allowed to be accessed in the "
        f"computation of the apodization function '{function.__name__}'.\n\n"
        f"Any handling of '{variable_name}' is already done by the decorator "
        f"\033[1m'as_apodization_function'\033[0m.\n\n"
        f"The following code accesses '{variable_name}':\n\n"
        f"File: {function_file_name}\n"
        f"Line: {offending_line_number_in_file}\n\n"
        f"{offending_code_lines}",
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
            x_max: RealNumeric = 1.0,  # ← mandatory default value
            *,  # ← mandatory enforcement of keyword-only arguments
            ...  # ← additional keyword arguments required by the apodization function
            skip_validation: bool = False,
        ) -> NDArray[np.float64]:
            ...
        ```

    Even though ``x_max`` is required to be the second argument, the decorator will
    internally scale the interval ``[-1, 1]`` to ``[-x_max, x_max]``. So, ``x_max``
    may not be used in the computation of the apodization function.

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
    ValueError
        If ``x_max`` is accessed in the computation of the apodization function.

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
            x_max: RealNumeric = 1.0, # ← mandatory default value
            *,  # ← mandatory enforcement of keyword-only arguments
            ...  # ← additional keyword arguments required by the apodization function
            skip_validation: bool = False,
        ) -> NDArray[np.float64]:
            ...
        ```

    Even though ``x_max`` is required to be the second argument, the decorator will
    internally scale the interval ``[-1, 1]`` to ``[-x_max, x_max]``. So, ``x_max``
    may not be used in the computation of the apodization function.

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

    # --- Signature Validation ---

    _validate_apodization_function_signature(function=apodization_function)

    # --- x_max Access Validation ---

    # this part is tricky because the function's source code is needed to check if
    # ``x_max`` is accessed in the computation of the apodization function
    _ensure_variable_is_not_used_in_function(
        function=apodization_function,
        variable_name="x_max",
    )

    # --- Wrapping ---

    wrapped_apodization_function = _convert_to_validated_apodization_function(
        apodization_function=apodization_function
    )

    # a tag is added to the wrapped function to indicate that it is a validated
    # apodization function
    wrapped_apodization_function.is_validated_apodization_function = True  # type: ignore

    return wrapped_apodization_function


def print_apodization_function_template(
    name: Optional[str] = None,
    with_imports: bool = True,
) -> None:
    """
    Prints a template for an apodization function to the console.
    This can simply be copied and pasted into a Python file. From there, the function
    can be implemented.

    As an example, a perfectly functional boxcar apodization function is already
    implemented. The template is designed to be as flexible as possible and to provide
    a good starting point for the implementation of an apodization function.

    Special remarks are underlined while the parts where custom code has to be inserted
    are bold.

    Parameters
    ----------
    name : :class:`str`, optional
        The name of the apodization function.
        If not provided, the name is set to ``"apodization_function"``.
    with_imports : :class:`bool`, default=``True``
        Whether to include the necessary imports for the template or not.

    """

    import_str = ""
    pyscopee_access_str = ""
    if with_imports:
        import_str = textwrap.dedent(
            """
                # === Imports ===

                import numpy as np
                from numpy.typing import NDArray

                import pyscopee as psc

                # === Function ===

            """
        )
        pyscopee_access_str = "psc."

    template = textwrap.dedent(
        """
            {import_str}def {name}(
                x: {pyscopee_access_str}RealNumericArrayLike,
                x_max: {pyscopee_access_str}RealNumeric = 1.0,
                *,
                \033[1m# Insert additional keyword-only arguments here.\033[0m
                skip_validation: bool = False,
            ) -> NDArray[np.float64]:
                \"\"\"
                \033[1mInsert the description of the apodization function here.\033[0m

                Parameters
                ----------
                x : Array-like of shape (n,)
                    The points at which to evaluate the apodization function.
                    Negative entries are converted to positive ones under the assumption that
                    the apodization function has even symmetry.
                    Its length has to be at least 1.
                    It is internally promoted to ``np.float64``.
                x_max : :class:`float` or :class:`int`, default=``1.0``
                    The maximum value of the x-range over which the apodization function is
                    defined.
                    It must be a positive real number ``> 0``.
                    With this, the x-range of ``[-1, 1]`` where apodization functions are
                    typically defined is scaled to ``[-x_max, x_max]``.

                \033[1mInsert additional keyword-only arguments here.\033[0m

                skip_validation : :class:`bool`, default=``False`` (keyword-only)
                    Whether to skip the input validation of ``x_max`` (``True``) or not
                    (``False``).
                    ``x`` is always validated.
                    This variable is meant for internal use only and it is highly
                    discouraged to set it to ``True``.

                Returns
                -------
                apodization_values : :class:`numpy.ndarray` of shape (n,) of dtype ``np.float64``
                    The values of the apodization function at the given points.

                \"\"\"  # noqa: E501

                \033[4m# It is not allowed to access ``x_max`` in the computation of the apodization\033[0m
                \033[4m# function. The decorator will handle ``x_max`` before the function is even\033[0m
                \033[4m# called.\033[0m

                # --- Input Validation ---

                if not skip_validation:
                    \033[1m# Insert input validation here.\033[0m
                    \033[1m# The decorator already validates ``x`` and ``x_max``.\033[0m
                    pass

                # --- Computation ---

                \033[1m# Insert computation here.\033[0m

                \033[1m# The apodization function only needs to be computed on the interval [0, 1]
                # where x = 0 is the center of the apodization and x = 1 is the right boundary
                # where it fades out (not necessarily to zero). Symmetry implies that x = -1 is
                # the left boundary analogous to x = 1.
                # Outside this interval, the apodization function will be zeroed automatically
                # independent of what this function would return.
                # The decorator will handle the symmetry f(-x) = f(x) and the scaling to
                # [-x_max, x_max].\033[0m

                \033[1m# The following is a dummy return statement that results in a boxcar
                # apodization\033[0m
                \033[1mreturn np.ones_like(x, dtype=np.float64)\033[0m

            """  # noqa: E501
    )

    print(
        template.format(
            name=name if name is not None else "apodization_function",
            import_str=import_str,
            pyscopee_access_str=pyscopee_access_str,
        )
    )

    return


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

    References
    ----------
    .. [1] Parker K. J., Apodization and Windowing Functions,
       Transactions on Ultrasonics, Ferroelectrics, and Frequency Control,
       Volume 60, Issue 6, 2013, pp. 1263 - 1271, DOI: 10.1109/TUFFC.2013.2691

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
