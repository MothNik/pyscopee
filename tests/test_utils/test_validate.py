"""
This test suite implements all tests for the module :mod:`pyscopee._utils.validate`.

"""

# === Imports ===

from array import array
from math import isclose as pyisclose
from typing import Any, Optional, Type, Union

import numpy as np
import pandas as pd
import pytest

from pyscopee._utils import (
    get_validated_integer,
    get_validated_real_numeric,
    get_validated_real_numeric_1d_array_like,
)

# === Tests ===


@pytest.mark.parametrize(
    "value, min_value, max_value, clip, expected",
    [
        (  # 0) a Python integer without any constraints
            1,
            None,
            None,
            False,
            1,
        ),
        (  # 1) a NumPy integer without any constraints
            np.int64(1),
            None,
            None,
            False,
            1,
        ),
        (  # 2) a float value without any constraints
            1.0,
            None,
            None,
            False,
            TypeError(
                "Expected 'value' to be of type 'int' / 'numpy.integer', but got "
                "'float'."
            ),
        ),
        (  # 3) a Python integer with a minimum constraint that is satisfied
            1,
            0,
            None,
            False,
            1,
        ),
        (  # 4) a Python integer with a minimum constraint that leads to clipping
            1,
            2,
            None,
            True,
            2,
        ),
        (  # 5) a Python integer with a minimum constraint that leads to an exception
            1,
            2,
            None,
            False,
            ValueError("Expected 'value' to be >= 2, but got 1."),
        ),
        (  # 6) a Python integer with a maximum constraint that is satisfied
            1,
            None,
            2,
            False,
            1,
        ),
        (  # 7) a Python integer with a maximum constraint that leads to clipping
            3,
            None,
            2,
            True,
            2,
        ),
        (  # 8) a Python integer with a maximum constraint that leads to an exception
            3,
            None,
            2,
            False,
            ValueError("Expected 'value' to be <= 2, but got 3."),
        ),
        (  # 9) a Python integer with both constraints that are satisfied
            1,
            0,
            2,
            False,
            1,
        ),
        (  # 10) a Python integer with both constraints that lead to clipping
            0,
            1,
            2,
            True,
            1,
        ),
        (  # 11) a Python integer with both constraints that lead to clipping
            3,
            0,
            2,
            True,
            2,
        ),
        (  # 12) a Python integer with both constraints that lead to an exception
            0,
            1,
            2,
            False,
            ValueError("Expected 'value' to be >= 1, but got 0."),
        ),
        (  # 13) a Python integer with both constraints that lead to an exception
            3,
            0,
            2,
            False,
            ValueError("Expected 'value' to be <= 2, but got 3."),
        ),
        (  # 14) a Python integer with flipped constraints
            1,
            2,
            0,
            False,
            ValueError(
                "Expected minimum value for 'value' to be <= maximum value, but got "
                "min = 2 and max = 0."
            ),
        ),
        (  # 15) a NumPy integer with a minimum constraint that is satisfied
            np.int64(1),
            0,
            None,
            False,
            1,
        ),
        (  # 16) a NumPy integer with a minimum constraint that leads to clipping
            np.int64(1),
            2,
            None,
            True,
            2,
        ),
        (  # 17) a NumPy integer with a minimum constraint that leads to an exception
            np.int64(1),
            2,
            None,
            False,
            ValueError("Expected 'value' to be >= 2, but got 1."),
        ),
        (  # 18) a NumPy integer with a maximum constraint that is satisfied
            np.int64(1),
            None,
            2,
            False,
            1,
        ),
        (  # 19) a NumPy integer with a maximum constraint that leads to clipping
            np.int64(3),
            None,
            2,
            True,
            2,
        ),
        (  # 20) a NumPy integer with a maximum constraint that leads to an exception
            np.int64(3),
            None,
            2,
            False,
            ValueError("Expected 'value' to be <= 2, but got 3."),
        ),
        (  # 21) a NumPy integer with both constraints that are satisfied
            np.int64(1),
            0,
            2,
            False,
            1,
        ),
        (  # 22) a NumPy integer with both constraints that lead to clipping
            np.int64(0),
            1,
            2,
            True,
            1,
        ),
        (  # 23) a NumPy integer with both constraints that lead to clipping
            np.int64(3),
            0,
            2,
            True,
            2,
        ),
        (  # 24) a NumPy integer with both constraints that lead to an exception
            np.int64(0),
            1,
            2,
            False,
            ValueError("Expected 'value' to be >= 1, but got 0."),
        ),
        (  # 25) a NumPy integer with both constraints that lead to an exception
            np.int64(3),
            0,
            2,
            False,
            ValueError("Expected 'value' to be <= 2, but got 3."),
        ),
        (  # 26) a NumPy integer with flipped constraints
            np.int64(1),
            2,
            0,
            False,
            ValueError(
                "Expected minimum value for 'value' to be <= maximum value, but got "
                "min = 2 and max = 0."
            ),
        ),
        (  # 27) a Python float with a minimum constraint that is satisfied
            1.0,
            0,
            None,
            False,
            TypeError(
                "Expected 'value' to be of type 'int' / 'numpy.integer', but got "
                "'float'."
            ),
        ),
        (  # 28) a Python float with a maximum constraint that is satisfied
            1.0,
            None,
            2,
            False,
            TypeError(
                "Expected 'value' to be of type 'int' / 'numpy.integer', but got "
                "'float'."
            ),
        ),
        (  # 29) a Python float with both constraints that are satisfied
            1.0,
            0,
            2,
            False,
            TypeError(
                "Expected 'value' to be of type 'int' / 'numpy.integer', but got "
                "'float'."
            ),
        ),
        (  # 30) a Python float with flipped constraints
            1.0,
            2,
            0,
            False,
            TypeError(
                "Expected 'value' to be of type 'int' / 'numpy.integer', but got "
                "'float'."
            ),
        ),
    ],
)
def test_integer_validation(
    value: Any,
    min_value: Optional[int],
    max_value: Optional[int],
    clip: bool,
    expected: Union[int, Exception],
) -> None:
    """
    Tests the function :func:`get_validated_integer` for various input values for

    - passing for correct input values
    - raising exceptions for incorrect input values

    """

    # if an exception should be raised, the function is called and the exception is
    # checked
    if isinstance(expected, Exception):
        with pytest.raises(type(expected), match=str(expected)):
            checked_value = get_validated_integer(
                value=value,
                name="value",
                min_value=min_value,
                max_value=max_value,
                clip=clip,
            )

        return

    # if no exception should be raised, the function is called and the output is checked
    checked_value = get_validated_integer(
        value=value,
        name="value",
        min_value=min_value,
        max_value=max_value,
        clip=clip,
    )

    assert checked_value == expected

    return


@pytest.mark.parametrize(
    "value, min_value, max_value, clip, expected",
    [
        (  # 0) a Python float without any constraints
            1.0,
            None,
            None,
            False,
            1.0,
        ),
        (  # 1) a NumPy float without any constraints
            np.float64(1.0),
            None,
            None,
            False,
            1.0,
        ),
        (  # 2) a Python integer without any constraints
            1,
            None,
            None,
            False,
            1.0,
        ),
        (  # 3) a NumPy integer without any constraints
            np.int64(1),
            None,
            None,
            False,
            1.0,
        ),
        (  # 4) a complex value without any constraints
            complex(1.0, 1.0),
            None,
            None,
            False,
            TypeError(
                "Expected 'value' to be of type 'float' / 'numpy.floating' / 'int' / "
                "'numpy.integer', but got 'complex'."
            ),
        ),
        (  # 5) a Python float with a minimum constraint that is satisfied
            1.0,
            0.0,
            None,
            False,
            1.0,
        ),
        (  # 6) a Python float with a minimum constraint that leads to clipping
            1.0,
            2.0,
            None,
            True,
            2.0,
        ),
        (  # 7) a Python float with a minimum constraint that leads to an exception
            1.0,
            2.0,
            None,
            False,
            ValueError("Expected 'value' to be >= 2.0, but got 1.0."),
        ),
        (  # 8) a Python integer with a minimum constraint that is satisfied
            1,
            0.0,
            None,
            False,
            1.0,
        ),
        (  # 9) a Python integer with a minimum constraint that leads to clipping
            1,
            2.0,
            None,
            True,
            2.0,
        ),
        (  # 10) a Python integer with a minimum constraint that leads to an exception
            1,
            2.0,
            None,
            False,
            ValueError("Expected 'value' to be >= 2.0, but got 1."),
        ),
        (  # 11) a Python float with a maximum constraint that is satisfied
            1.0,
            None,
            2.0,
            False,
            1.0,
        ),
        (  # 12) a Python float with a maximum constraint that leads to clipping
            3.0,
            None,
            2.0,
            True,
            2.0,
        ),
        (  # 13) a Python float with a maximum constraint that leads to an exception
            3.0,
            None,
            2.0,
            False,
            ValueError("Expected 'value' to be <= 2.0, but got 3.0."),
        ),
        (  # 14) a Python integer with a maximum constraint that is satisfied
            1,
            None,
            2.0,
            False,
            1.0,
        ),
        (  # 15) a Python integer with a maximum constraint that leads to clipping
            3,
            None,
            2.0,
            True,
            2.0,
        ),
        (  # 16) a Python integer with a maximum constraint that leads to an exception
            3,
            None,
            2.0,
            False,
            ValueError("Expected 'value' to be <= 2.0, but got 3."),
        ),
        (  # 17) a Python float with both constraints that are satisfied
            1.0,
            0.0,
            2.0,
            False,
            1.0,
        ),
        (  # 18) a Python float with both constraints that lead to clipping
            0.0,
            1.0,
            2.0,
            True,
            1.0,
        ),
        (  # 19) a Python float with both constraints that lead to clipping
            3.0,
            0.0,
            2.0,
            True,
            2.0,
        ),
        (  # 20) a Python float with both constraints that lead to an exception
            0.0,
            1.0,
            2.0,
            False,
            ValueError("Expected 'value' to be >= 1.0, but got 0.0."),
        ),
        (  # 21) a Python float with both constraints that lead to an exception
            3.0,
            0.0,
            2.0,
            False,
            ValueError("Expected 'value' to be <= 2.0, but got 3.0."),
        ),
        (  # 22) a Python integer with both constraints that are satisfied
            1,
            0.0,
            2.0,
            False,
            1.0,
        ),
        (  # 23) a Python integer with both constraints that lead to clipping
            0,
            1.0,
            2.0,
            True,
            1.0,
        ),
        (  # 24) a Python integer with both constraints that lead to clipping
            3,
            0.0,
            2.0,
            True,
            2.0,
        ),
        (  # 25) a Python integer with both constraints that lead to an exception
            0,
            1.0,
            2.0,
            False,
            ValueError("Expected 'value' to be >= 1.0, but got 0."),
        ),
        (  # 26) a Python integer with both constraints that lead to an exception
            3,
            0.0,
            2.0,
            False,
            ValueError("Expected 'value' to be <= 2.0, but got 3."),
        ),
        (  # 27) a Python float with flipped constraints
            1.0,
            2.0,
            0.0,
            False,
            ValueError(
                "Expected minimum value for 'value' to be <= maximum value, but got "
                "min = 2.0 and max = 0.0."
            ),
        ),
        (  # 28) a Python integer with flipped constraints
            1,
            2.0,
            0.0,
            False,
            ValueError(
                "Expected minimum value for 'value' to be <= maximum value, but got "
                "min = 2.0 and max = 0.0."
            ),
        ),
        (  # 29) a NumPy float with a minimum constraint that is satisfied
            np.float64(1.0),
            0.0,
            None,
            False,
            1.0,
        ),
        (  # 30) a NumPy float with a minimum constraint that leads to clipping
            np.float64(1.0),
            2.0,
            None,
            True,
            2.0,
        ),
        (  # 31) a NumPy float with a minimum constraint that leads to an exception
            np.float64(1.0),
            2.0,
            None,
            False,
            ValueError("Expected 'value' to be >= 2.0, but got 1.0."),
        ),
        (  # 32) a NumPy float with a maximum constraint that is satisfied
            np.float64(1.0),
            None,
            2.0,
            False,
            1.0,
        ),
        (  # 33) a NumPy float with a maximum constraint that leads to clipping
            np.float64(3.0),
            None,
            2.0,
            True,
            2.0,
        ),
        (  # 34) a NumPy float with a maximum constraint that leads to an exception
            np.float64(3.0),
            None,
            2.0,
            False,
            ValueError("Expected 'value' to be <= 2.0, but got 3.0."),
        ),
        (  # 35) a NumPy float with both constraints that are satisfied
            np.float64(1.0),
            0.0,
            2.0,
            False,
            1.0,
        ),
        (  # 36) a NumPy float with both constraints that lead to clipping
            np.float64(0.0),
            1.0,
            2.0,
            True,
            1.0,
        ),
        (  # 37) a NumPy float with both constraints that lead to clipping
            np.float64(3.0),
            0.0,
            2.0,
            True,
            2.0,
        ),
        (  # 38) a NumPy float with both constraints that lead to an exception
            np.float64(0.0),
            1.0,
            2.0,
            False,
            ValueError("Expected 'value' to be >= 1.0, but got 0.0."),
        ),
        (  # 39) a NumPy float with both constraints that lead to an exception
            np.float64(3.0),
            0.0,
            2.0,
            False,
            ValueError("Expected 'value' to be <= 2.0, but got 3.0."),
        ),
        (  # 40) a NumPy float with flipped constraints
            np.float64(1.0),
            2.0,
            0.0,
            False,
            ValueError(
                "Expected minimum value for 'value' to be <= maximum value, but got "
                "min = 2.0 and max = 0.0."
            ),
        ),
        (  # 41) a NumPy integer with a minimum constraint that is satisfied
            np.int64(1),
            0,
            None,
            False,
            1.0,
        ),
        (  # 42) a NumPy integer with a minimum constraint that leads to clipping
            np.int64(1),
            2,
            None,
            True,
            2.0,
        ),
        (  # 43) a NumPy integer with a minimum constraint that leads to an exception
            np.int64(1),
            2,
            None,
            False,
            ValueError("Expected 'value' to be >= 2, but got 1."),
        ),
        (  # 44) a NumPy integer with a maximum constraint that is satisfied
            np.int64(1),
            None,
            2,
            False,
            1.0,
        ),
        (  # 45) a NumPy integer with a maximum constraint that leads to clipping
            np.int64(3),
            None,
            2,
            True,
            2.0,
        ),
        (  # 46) a NumPy integer with a maximum constraint that leads to an exception
            np.int64(3),
            None,
            2,
            False,
            ValueError("Expected 'value' to be <= 2, but got 3."),
        ),
        (  # 47) a NumPy integer with both constraints that are satisfied
            np.int64(1),
            0,
            2,
            False,
            1.0,
        ),
        (  # 48) a NumPy integer with both constraints that lead to clipping
            np.int64(0),
            1,
            2,
            True,
            1.0,
        ),
        (  # 49) a NumPy integer with both constraints that lead to clipping
            np.int64(3),
            0,
            2,
            True,
            2.0,
        ),
        (  # 50) a NumPy integer with both constraints that lead to an exception
            np.int64(0),
            1,
            2,
            False,
            ValueError("Expected 'value' to be >= 1, but got 0."),
        ),
        (  # 51) a NumPy integer with both constraints that lead to an exception
            np.int64(3),
            0,
            2,
            False,
            ValueError("Expected 'value' to be <= 2, but got 3."),
        ),
        (  # 52) a NumPy integer with flipped constraints
            np.int64(1),
            2,
            0,
            False,
            ValueError(
                "Expected minimum value for 'value' to be <= maximum value, but got "
                "min = 2 and max = 0."
            ),
        ),
        (  # 53) a complex value with a minimum constraint that is satisfied
            complex(1.0, 1.0),
            0.0,
            None,
            False,
            TypeError(
                "Expected 'value' to be of type 'float' / 'numpy.floating' / 'int' / "
                "'numpy.integer', but got 'complex'."
            ),
        ),
        (  # 54) a complex value with a maximum constraint that is satisfied
            complex(1.0, 1.0),
            None,
            2.0,
            False,
            TypeError(
                "Expected 'value' to be of type 'float' / 'numpy.floating' / 'int' / "
                "'numpy.integer', but got 'complex'."
            ),
        ),
        (  # 55) a complex value with both constraints that are satisfied
            complex(1.0, 1.0),
            0.0,
            2.0,
            False,
            TypeError(
                "Expected 'value' to be of type 'float' / 'numpy.floating' / 'int' / "
                "'numpy.integer', but got 'complex'."
            ),
        ),
        (  # 56) a complex value with flipped constraints
            complex(1.0, 1.0),
            2.0,
            0.0,
            False,
            TypeError(
                "Expected 'value' to be of type 'float' / 'numpy.floating' / 'int' / "
                "'numpy.integer', but got 'complex'."
            ),
        ),
    ],
)
def test_real_numeric_validation(
    value: Any,
    min_value: Optional[float],
    max_value: Optional[float],
    clip: bool,
    expected: Union[float, Exception],
) -> None:
    """
    Tests the function :func:`get_validated_real_numeric` for various input values for

    - passing for correct input values
    - raising exceptions for incorrect input values

    """

    # if an exception should be raised, the function is called and the exception is
    # checked
    if isinstance(expected, Exception):
        with pytest.raises(type(expected), match=str(expected)):
            checked_value = get_validated_real_numeric(
                value=value,
                name="value",
                min_value=min_value,
                max_value=max_value,
                clip=clip,
            )

        return

    # if no exception should be raised, the function is called and the output is checked
    checked_value = get_validated_real_numeric(
        value=value,
        name="value",
        min_value=min_value,
        max_value=max_value,
        clip=clip,
    )

    assert pyisclose(
        checked_value,
        expected,
        abs_tol=1e-15,
        rel_tol=1e-15,
    )

    return


@pytest.mark.parametrize(
    "value, min_size, max_size, output_dtype, expected",
    [
        (  # 0) a NumPy Array without any constraints
            np.array([1.0]),
            None,
            None,
            None,
            np.array([1.0]),
        ),
        (  # 1) a NumPy Array with a satisfied minimum size
            np.array([1.0]),
            1,
            None,
            None,
            np.array([1.0]),
        ),
        (  # 2) a NumPy Array with a violated minimum size
            np.array([1.0]),
            2,
            None,
            None,
            ValueError(
                "Expected 'value' to have a size between 2 and None for axis 0, but "
                "got a size of 1."
            ),
        ),
        (  # 3) a NumPy Array with a satisfied maximum size
            np.array([1.0]),
            None,
            1,
            None,
            np.array([1.0]),
        ),
        (  # 4) a NumPy Array with a violated maximum size
            np.array([1.0]),
            None,
            0,
            None,
            ValueError(
                "Expected 'value' to have a size between None and 0 for axis 0, but "
                "got a size of 1."
            ),
        ),
        (  # 5) a NumPy Array with a satisfied fixed size
            np.array([1.0]),
            1,
            1,
            None,
            np.array([1.0]),
        ),
        (  # 6) a NumPy Array with a violated fixed size
            np.array([1.0]),
            2,
            2,
            None,
            ValueError(
                "Expected 'value' to have a size between 2 and 2 for axis 0, but got "
                "a size of 1."
            ),
        ),
        (  # 7) a NumPy Array with a satisfied minimum size and a violated maximum size
            np.array([1.0]),
            1,
            0,
            None,
            ValueError(
                "Expected 'value' to have a size between 1 and 0 for axis 0, but got "
                "a size of 1."
            ),
        ),
        (  # 8) a NumPy Array with a violated minimum size and a satisfied maximum size
            np.array([1.0]),
            2,
            1,
            None,
            ValueError(
                "Expected 'value' to have a size between 2 and 1 for axis 0, but got "
                "a size of 1."
            ),
        ),
        (  # 9) a NumPy Array with a satisfied minimum size and a satisfied maximum size
            np.array([1.0]),
            1,
            1,
            None,
            np.array([1.0]),
        ),
        (  # 10) a NumPy Array with a violated minimum size and a violated maximum size
            np.array([1.0]),
            2,
            0,
            None,
            ValueError(
                "Expected 'value' to have a size between 2 and 0 for axis 0, but got "
                "a size of 1."
            ),
        ),
        (  # 11) a NumPy Array with a valid type conversion
            np.array([1.0], dtype=np.float32),
            None,
            None,
            np.float64,
            np.array([1.0], dtype=np.float64),
        ),
        (  # 12) a NumPy Array with an invalid type conversion
            np.array([1], dtype=np.float32),
            None,
            None,
            np.int64,
            # NOTE: raw string with bracket escape for preventing regex errors
            #       https://stackoverflow.com/a/76565993/14814813
            TypeError(
                r"Could not convert 'value' from a 'float32'- to a 'int64'-Array "
                r"\(uses 'safe' casting\)."
            ),
        ),
        (  # 13) an empty NumPy Array without any constraints
            np.array([]),
            None,
            None,
            None,
            ValueError("Expected 'value' to be a non-empty Array-like."),
        ),
        (  # 14) a 2D NumPy Array without any constraints
            np.array([[1.0]]),
            None,
            None,
            None,
            # NOTE: raw string with bracket escape for preventing regex errors
            #       https://stackoverflow.com/a/76565993/14814813
            ValueError(
                r"Expected 'value' to be a 1D Array-like, but got a 2D Array-like of "
                r"shape \(1, 1\)."
            ),
        ),
        (  # 15) a Python List without any constraints
            [1.0],
            None,
            None,
            None,
            np.array([1.0]),
        ),
        (  # 16) a Python List with a satisfied minimum size
            [1.0],
            1,
            None,
            None,
            np.array([1.0]),
        ),
        (  # 17) a Python List with a violated minimum size
            [1.0],
            2,
            None,
            None,
            ValueError(
                "Expected 'value' to have a size between 2 and None for axis 0, but "
                "got a size of 1."
            ),
        ),
        (  # 18) a Python List with a satisfied maximum size
            [1.0],
            None,
            1,
            None,
            np.array([1.0]),
        ),
        (  # 19) a Python List with a violated maximum size
            [1.0],
            None,
            0,
            None,
            ValueError(
                "Expected 'value' to have a size between None and 0 for axis 0, but "
                "got a size of 1."
            ),
        ),
        (  # 20) a Python List with a satisfied fixed size
            [1.0],
            1,
            1,
            None,
            np.array([1.0]),
        ),
        (  # 21) a Python List with a violated fixed size
            [1.0],
            2,
            2,
            None,
            ValueError(
                "Expected 'value' to have a size between 2 and 2 for axis 0, but got "
                "a size of 1."
            ),
        ),
        (  # 22) a Python List with a satisfied minimum size and a violated maximum size
            [1.0],
            1,
            0,
            None,
            ValueError(
                "Expected 'value' to have a size between 1 and 0 for axis 0, but got "
                "a size of 1."
            ),
        ),
        (  # 23) a Python List with a violated minimum size and a satisfied maximum size
            [1.0],
            2,
            1,
            None,
            ValueError(
                "Expected 'value' to have a size between 2 and 1 for axis 0, but got "
                "a size of 1."
            ),
        ),
        (  # 24) a Python List with a satisfied minimum size and a satisfied maximum size  # noqa: E501
            [1.0],
            1,
            1,
            None,
            np.array([1.0]),
        ),
        (  # 25) a Python List with a violated minimum size and a violated maximum size
            [1.0],
            2,
            0,
            None,
            ValueError(
                "Expected 'value' to have a size between 2 and 0 for axis 0, but got "
                "a size of 1."
            ),
        ),
        (  # 26) a Python List with a valid type conversion
            [1.0],
            None,
            None,
            np.float64,
            np.float64(1.0),
        ),
        (  # 27) a Python List with an invalid type conversion
            [1.0],
            None,
            None,
            np.int64,
            # NOTE: raw string with bracket escape for preventing regex errors
            #       https://stackoverflow.com/a/76565993/14814813
            TypeError(
                r"Could not convert 'value' from a 'float64'- to a 'int64'-Array "
                r"\(uses 'safe' casting\)."
            ),
        ),
        (  # 28) an empty Python List without any constraints
            [],
            None,
            None,
            None,
            ValueError("Expected 'value' to be a non-empty Array-like."),
        ),
        (  # 29) a 2D Python List without any constraints
            [[1.0]],
            None,
            None,
            None,
            # NOTE: raw string with bracket escape for preventing regex errors
            #       https://stackoverflow.com/a/76565993/14814813
            ValueError(
                r"Expected 'value' to be a 1D Array-like, but got a 2D Array-like of "
                r"shape \(1, 1\)."
            ),
        ),
        (  # 30) a Python Tuple without any constraints
            (1.0,),
            None,
            None,
            None,
            np.array([1.0]),
        ),
        (  # 31) a Python Tuple with a satisfied minimum size
            (1.0,),
            1,
            None,
            None,
            np.array([1.0]),
        ),
        (  # 32) a Python Tuple with a violated minimum size
            (1.0,),
            2,
            None,
            None,
            ValueError(
                "Expected 'value' to have a size between 2 and None for axis 0, but "
                "got a size of 1."
            ),
        ),
        (  # 33) a Python Tuple with a satisfied maximum size
            (1.0,),
            None,
            1,
            None,
            np.array([1.0]),
        ),
        (  # 34) a Python Tuple with a violated maximum size
            (1.0,),
            None,
            0,
            None,
            ValueError(
                "Expected 'value' to have a size between None and 0 for axis 0, but "
                "got a size of 1."
            ),
        ),
        (  # 35) a Python Tuple with a satisfied fixed size
            (1.0,),
            1,
            1,
            None,
            np.array([1.0]),
        ),
        (  # 36) a Python Tuple with a violated fixed size
            (1.0,),
            2,
            2,
            None,
            ValueError(
                "Expected 'value' to have a size between 2 and 2 for axis 0, but got "
                "a size of 1."
            ),
        ),
        (  # 37) a Python Tuple with a satisfied minimum size and a violated maximum size  # noqa: E501
            (1.0,),
            1,
            0,
            None,
            ValueError(
                "Expected 'value' to have a size between 1 and 0 for axis 0, but got "
                "a size of 1."
            ),
        ),
        (  # 38) a Python Tuple with a violated minimum size and a satisfied maximum size  # noqa: E501
            (1.0,),
            2,
            1,
            None,
            ValueError(
                "Expected 'value' to have a size between 2 and 1 for axis 0, but got "
                "a size of 1."
            ),
        ),
        (  # 39) a Python Tuple with a satisfied minimum size and a satisfied maximum size  # noqa: E501
            (1.0,),
            1,
            1,
            None,
            np.array([1.0]),
        ),
        (  # 40) a Python Tuple with a violated minimum size and a violated maximum size
            (1.0,),
            2,
            0,
            None,
            ValueError(
                "Expected 'value' to have a size between 2 and 0 for axis 0, but got "
                "a size of 1."
            ),
        ),
        (  # 41) a Python Tuple with a valid type conversion
            (1.0,),
            None,
            None,
            np.float64,
            np.float64(1.0),
        ),
        (  # 42) a Python Tuple with an invalid type conversion
            (1.0,),
            None,
            None,
            np.int64,
            # NOTE: raw string with bracket escape for preventing regex errors
            #       https://stackoverflow.com/a/76565993/14814813
            TypeError(
                r"Could not convert 'value' from a 'float64'- to a 'int64'-Array "
                r"\(uses 'safe' casting\)."
            ),
        ),
        (  # 43) an empty Python Tuple without any constraints
            tuple(),
            None,
            None,
            None,
            ValueError("Expected 'value' to be a non-empty Array-like."),
        ),
        (  # 44) a 2D Python Tuple without any constraints
            ((1.0,),),
            None,
            None,
            None,
            # NOTE: raw string with bracket escape for preventing regex errors
            #       https://stackoverflow.com/a/76565993/14814813
            ValueError(
                r"Expected 'value' to be a 1D Array-like, but got a 2D Array-like of "
                r"shape \(1, 1\)."
            ),
        ),
        (  # 45) a Python Array without any constraints
            array("d", [1.0]),
            None,
            None,
            None,
            np.array([1.0]),
        ),
        (  # 46) a Python Array with a satisfied minimum size
            array("d", [1.0]),
            1,
            None,
            None,
            np.array([1.0]),
        ),
        (  # 47) a Python Array with a violated minimum size
            array("d", [1.0]),
            2,
            None,
            None,
            ValueError(
                "Expected 'value' to have a size between 2 and None for axis 0, but "
                "got a size of 1."
            ),
        ),
        (  # 48) a Python Array with a satisfied maximum size
            array("d", [1.0]),
            None,
            1,
            None,
            np.array([1.0]),
        ),
        (  # 49) a Python Array with a violated maximum size
            array("d", [1.0]),
            None,
            0,
            None,
            ValueError(
                "Expected 'value' to have a size between None and 0 for axis 0, but "
                "got a size of 1."
            ),
        ),
        (  # 50) a Python Array with a satisfied fixed size
            array("d", [1.0]),
            1,
            1,
            None,
            np.array([1.0]),
        ),
        (  # 51) a Python Array with a violated fixed size
            array("d", [1.0]),
            2,
            2,
            None,
            ValueError(
                "Expected 'value' to have a size between 2 and 2 for axis 0, but got "
                "a size of 1."
            ),
        ),
        (  # 52) a Python Array with a satisfied minimum size and a violated maximum size  # noqa: E501
            array("d", [1.0]),
            1,
            0,
            None,
            ValueError(
                "Expected 'value' to have a size between 1 and 0 for axis 0, but got "
                "a size of 1."
            ),
        ),
        (  # 53) a Python Array with a violated minimum size and a satisfied maximum size # noqa: E501
            array("d", [1.0]),
            2,
            1,
            None,
            ValueError(
                "Expected 'value' to have a size between 2 and 1 for axis 0, but got "
                "a size of 1."
            ),
        ),
        (  # 54) a Python Array with a satisfied minimum size and a satisfied maximum size  # noqa: E501
            array("d", [1.0]),
            1,
            1,
            None,
            np.array([1.0]),
        ),
        (  # 55) a Python Array with a violated minimum size and a violated maximum size
            array("d", [1.0]),
            2,
            0,
            None,
            ValueError(
                "Expected 'value' to have a size between 2 and 0 for axis 0, but got "
                "a size of 1."
            ),
        ),
        (  # 56) a Python Array with a valid type conversion
            array("d", [1.0]),
            None,
            None,
            np.float64,
            np.float64(1.0),
        ),
        (  # 57) a Python Array with an invalid type conversion
            array("d", [1.0]),
            None,
            None,
            np.int64,
            # NOTE: raw string with bracket escape for preventing regex errors
            #       https://stackoverflow.com/a/76565993/14814813
            TypeError(
                r"Could not convert 'value' from a 'float64'- to a 'int64'-Array "
                r"\(uses 'safe' casting\)."
            ),
        ),
        (  # 58) an empty Python Array without any constraints
            array("d", []),
            None,
            None,
            None,
            ValueError("Expected 'value' to be a non-empty Array-like."),
        ),
        (  # 59) a "2D Python Array" without any constraints
            [array("d", [1.0])],
            None,
            None,
            None,
            # NOTE: raw string with bracket escape for preventing regex errors
            #       https://stackoverflow.com/a/76565993/14814813
            ValueError(
                r"Expected 'value' to be a 1D Array-like, but got a 2D Array-like of "
                r"shape \(1, 1\)."
            ),
        ),
        (  # 60) a Pandas Series without any constraints
            pd.Series([1.0]),
            None,
            None,
            None,
            np.array([1.0]),
        ),
        (  # 61) a Pandas Series with a satisfied minimum size
            pd.Series([1.0]),
            1,
            None,
            None,
            np.array([1.0]),
        ),
        (  # 62) a Pandas Series with a violated minimum size
            pd.Series([1.0]),
            2,
            None,
            None,
            ValueError(
                "Expected 'value' to have a size between 2 and None for axis 0, but "
                "got a size of 1."
            ),
        ),
        (  # 63) a Pandas Series with a satisfied maximum size
            pd.Series([1.0]),
            None,
            1,
            None,
            np.array([1.0]),
        ),
        (  # 64) a Pandas Series with a violated maximum size
            pd.Series([1.0]),
            None,
            0,
            None,
            ValueError(
                "Expected 'value' to have a size between None and 0 for axis 0, but "
                "got a size of 1."
            ),
        ),
        (  # 65) a Pandas Series with a satisfied fixed size
            pd.Series([1.0]),
            1,
            1,
            None,
            np.array([1.0]),
        ),
        (  # 66) a Pandas Series with a violated fixed size
            pd.Series([1.0]),
            2,
            2,
            None,
            ValueError(
                "Expected 'value' to have a size between 2 and 2 for axis 0, but got "
                "a size of 1."
            ),
        ),
        (  # 67) a Pandas Series with a satisfied minimum size and a violated maximum size  # noqa: E501
            pd.Series([1.0]),
            1,
            0,
            None,
            ValueError(
                "Expected 'value' to have a size between 1 and 0 for axis 0, but got "
                "a size of 1."
            ),
        ),
        (  # 68) a Pandas Series with a violated minimum size and a satisfied maximum size  # noqa: E501
            pd.Series([1.0]),
            2,
            1,
            None,
            ValueError(
                "Expected 'value' to have a size between 2 and 1 for axis 0, but got "
                "a size of 1."
            ),
        ),
        (  # 69) a Pandas Series with a satisfied minimum size and a satisfied maximum size  # noqa: E501
            pd.Series([1.0]),
            1,
            1,
            None,
            np.array([1.0]),
        ),
        (  # 70) a Pandas Series with a violated minimum size and a violated maximum size  # noqa: E501
            pd.Series([1.0]),
            2,
            0,
            None,
            ValueError(
                "Expected 'value' to have a size between 2 and 0 for axis 0, but got "
                "a size of 1."
            ),
        ),
        (  # 71) a Pandas Series with a valid type conversion
            pd.Series([1.0]),
            None,
            None,
            np.float64,
            np.float64(1.0),
        ),
        (  # 72) a Pandas Series with an invalid type conversion
            pd.Series([1.0]),
            None,
            None,
            np.int64,
            # NOTE: raw string with bracket escape for preventing regex errors
            #       https://stackoverflow.com/a/76565993/14814813
            TypeError(
                r"Could not convert 'value' from a 'float64'- to a 'int64'-Array "
                r"\(uses 'safe' casting\)."
            ),
        ),
        (  # 73) an empty Pandas Series without any constraints
            pd.Series([]),
            None,
            None,
            None,
            ValueError("Expected 'value' to be a non-empty Array-like."),
        ),
        (  # 74) a "2D Pandas Series" without any constraints
            pd.DataFrame([1.0]),
            None,
            None,
            None,
            # NOTE: raw string with bracket escape for preventing regex errors
            #       https://stackoverflow.com/a/76565993/14814813
            ValueError(
                r"Expected 'value' to be a 1D Array-like, but got a 2D Array-like of "
                r"shape \(1, 1\)."
            ),
        ),
        (  # 75) a Python lists of lists with inconsistent sizes
            [[1.0], [2.0, 3.0]],
            None,
            None,
            None,
            ValueError("'value' could not be converted to a NumPy Array-like."),
        ),
        (  # 76) a NumPy Array of complex values
            np.array([1.0 + 1.0j]),
            None,
            None,
            None,
            TypeError(
                "Expected 'value' to be a 1D Array-like of real numeric values, but "
                "got a 1D Array-like with non-real numeric values."
            ),
        ),
        (  # 77) a Python List of complex values
            [1.0 + 1.0j],
            None,
            None,
            None,
            TypeError(
                "Expected 'value' to be a 1D Array-like of real numeric values, but "
                "got a 1D Array-like with non-real numeric values."
            ),
        ),
        (  # 78) a Python Tuple of complex values
            (1.0 + 1.0j,),
            None,
            None,
            None,
            TypeError(
                "Expected 'value' to be a 1D Array-like of real numeric values, but "
                "got a 1D Array-like with non-real numeric values."
            ),
        ),
        (  # 79) a Pandas Series of complex values
            pd.Series([1.0 + 1.0j]),
            None,
            None,
            None,
            TypeError(
                "Expected 'value' to be a 1D Array-like of real numeric values, but "
                "got a 1D Array-like with non-real numeric values."
            ),
        ),
    ],
)
def test_real_numeric_1d_array_like_validation(
    value: Any,
    min_size: Optional[int],
    max_size: Optional[int],
    output_dtype: Optional[Type],
    expected: Union[np.ndarray, Exception],
) -> None:
    """
    Tests the function :func:`get_validated_real_numeric_1d_array_like` for various
    input values for

    - passing for correct input values
    - raising exceptions for incorrect input values

    """

    # if an exception should be raised, the function is called and the exception is
    # checked
    if isinstance(expected, Exception):
        with pytest.raises(type(expected), match=str(expected)):
            checked_value = get_validated_real_numeric_1d_array_like(
                value=value,
                name="value",
                min_size=min_size,
                max_size=max_size,
                output_dtype=output_dtype,
            )

        return

    # if no exception should be raised, the function is called and the output is checked
    checked_value = get_validated_real_numeric_1d_array_like(
        value=value,
        name="value",
        min_size=min_size,
        max_size=max_size,
        output_dtype=output_dtype,
    )

    if output_dtype is not None:
        assert checked_value.dtype == output_dtype
    elif isinstance(value, np.ndarray):
        assert checked_value.dtype == value.dtype

    assert np.allclose(
        checked_value,
        expected,
        atol=1e-15,
        rtol=1e-15,
    )

    return
