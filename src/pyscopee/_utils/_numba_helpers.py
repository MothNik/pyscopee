"""
Module :mod:`_utils.numba_helpers`

This module implements auxiliary functionalities to handle Numba-related tasks, such as

- checking whether Numba ``jit``-compilation has been explicitly specified to take no
    effect, e.g., for test coverage

"""

# === Setup ===

__all__ = ["jit"]

# === Imports ===

import os
from enum import Enum
from typing import Callable

# === Models ===

# an Enum that specifies the possible actions that can be taken regarding Numba
# ``jit``-compilation


class NumbaJitActions(Enum):
    """
    Specifies the possible actions that can be taken regarding Numba
    ``jit``-compilation.

    """

    NORMAL = "0"
    DEACTIVATE = "1"


# === Constants ===

# the runtime argument that is used to specify that Numba ``jit``-compilation should
# take no effect
NUMBA_NO_JIT_ARGV = "--no-jit"

# the environment variable that is used to specify that Numba ``jit``-compilation should
# take no effect
NUMBA_NO_JIT_ENV_KEY = "CUSTOM_NUMBA_NO_JIT"


# whether the environment variable is set to specify that Numba ``jit``-compilation
# should take effect or not in the current runtime environment
_do_numba_normal_jit_action = (
    os.environ.get(NUMBA_NO_JIT_ENV_KEY, NumbaJitActions.NORMAL.value)
    == NumbaJitActions.NORMAL.value
)

# if Numba is not available at runtime, then the environment variable has to be
# ignored
try:
    import numba as __numba

    _numba_available = True

except ImportError:
    _numba_available = False

_do_numba_normal_jit_action = _do_numba_normal_jit_action and _numba_available


# === Functions ===

# if Numba ``jit``-compilation can be used, the ``jit`` decorator is imported from Numba
if _do_numba_normal_jit_action:
    jit = __numba.jit  # type: ignore

# if Numba ``jit``-compilation cannot be used or was disabled, a fake decorator is
# defined to be able to use the same syntax in the code
else:

    def jit(*args, **kwargs) -> Callable:
        """
        Fake decorator that can be used to make sure that Numba ``jit``-compilation has
        no effect when Numba was not available or disabled.

        Parameters
        ----------
        func : callable
            The function that is decorated.
        args : :obj:`tuple`
            The fake positional arguments.
        kwargs : :obj:`dict`
            The fake keyword arguments.

        Returns
        -------
        decorated_func : callable
            The decorated function.

        """

        def decorator(func: Callable) -> Callable:
            return func

        return decorator
