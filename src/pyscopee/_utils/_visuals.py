"""
Module :mod:`_utils._visuals`

This module provides utility functions that assist in visualisation tasks across the
``pyscopee`` package.

"""

# === Setup ===

__all__ = [
    "get_pyscopee_style",
    "apply_pyscopee_plot_style",
]

# === Imports ===

from pathlib import Path
from typing import List, Union

from matplotlib import style as mplstyle

# === Constants ===

# the path to default style for plots in the `pyscopee` package (relative to this file)
_DEFAULT_STYLE_PATH = "pyscopee.mplstyle"


# === Functions ===


def get_pyscopee_style() -> Path:
    """
    Returns the path to the ``pyscopee`` style for plots.

    Returns
    -------
    style_path : :class:`pathlib.Path`
        The path to the ``pyscopee`` style for plots.

    """

    return Path(__file__).parent / _DEFAULT_STYLE_PATH


def apply_pyscopee_plot_style(other_styles: Union[str, List[str], None] = None):
    """
    Applies the default ``pyscopee`` style to plots together with optional additional
    styles.

    Parameters
    ----------
    other_styles : :class:`str`, [:class:`str`], or ``None``, default=``None``
        The name of the style or a list of style names to apply in addition to the
        default style for plots in the ``pyscopee`` package.
        If ``None``, only the default style is applied.

    """

    style_paths = [get_pyscopee_style()]
    if other_styles is not None:
        if isinstance(other_styles, str):
            other_styles = [other_styles]
        style_paths.extend(other_styles)  # type: ignore

    mplstyle.use(style_paths)

    return
