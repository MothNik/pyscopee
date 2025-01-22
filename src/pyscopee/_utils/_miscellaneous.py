"""
Module :mod:`_utils._miscellaneous`

This module provides a collection of miscellaneous utility functions that are used
across the ``pyscopee`` package.

"""

# === Setup ===

__all__ = [
    "split_class_name_to_readable",
    "warn_verbose",
]

# === Imports ===

import re
from warnings import warn

# === Functions ===


def split_class_name_to_readable(obj: object) -> str:
    """
    Splits a class name into a more readable format by inserting spaces between
    capital letters.

    Parameters
    ----------
    obj : :class:`object`
        The object whose class name is to be split.

    Returns
    -------
    readable_class_name : :class:`str`
        The class name with spaces inserted between capital letters.

    """

    return re.sub(r"(?<!^)(?=[A-Z])", " ", obj.__class__.__name__)


def warn_verbose(
    message: str,
    category: type,
    issue_warning: bool,
) -> None:
    """
    Issues a warning if requested.

    Parameters
    ----------
    message : :class:`str`
        The warning message.
    category : :class:`type`
        The warning category.
    issue_warning : :class:`bool`
        Whether to issue the warning (``True``) or not (``False``).

    """  # noqa: E501

    if issue_warning:
        warn(message, category)
        return

    return
