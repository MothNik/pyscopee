"""
Module :mod:`_utils._visuals`

This module provides utility functions that assist in visualisation tasks across the
``pyscopee`` package.

"""

# === Setup ===

__all__ = [
    "apply_pyscopee_plot_style",
    "get_pyscopee_style",
    "pyscopee_plot_style",
]

# === Imports ===

from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Union

from matplotlib import style as mplstyle

# === Constants ===

# the path to default style for plots in the `pyscopee` package (relative to this file)
_DEFAULT_STYLE_PATH = "pyscopee.mplstyle"


# === Types ===

PlotStylePaths = Union[str, Path, List[Union[str, Path]], None]

# === Functions ===


def get_pyscopee_style() -> Path:
    """
    Returns the path to the ``pyscopee`` style for plots.

    Returns
    -------
    style_path : :obj:`pathlib.Path`
        The path to the ``pyscopee`` style for plots.

    """

    return Path(__file__).parent / _DEFAULT_STYLE_PATH


def _concatenate_pyscopee_with_other_styles(
    other_styles: PlotStylePaths,
) -> List[Union[str, Path]]:
    """
    Concatenates the default ``pyscopee`` style with additional styles.

    Parameters
    ----------
    other_styles : :obj:`str`, :obj:`pathlib.Path`, [:obj:`str` or :obj:`pathlib.Path`], or ``None``, default=``None``
        The name of the style or a list of style names to apply in addition to the
        default style for plots in the ``pyscopee`` package.
        If ``None``, only the default style is applied.

    Returns
    -------
    style_paths : [:obj:`str` or :obj:`pathlib.Path`]
        The list of style paths to apply to plots.

    """  # noqa: E501

    style_paths = [get_pyscopee_style()]
    if other_styles is not None:
        if isinstance(other_styles, str):
            other_styles = [other_styles]
        style_paths.extend(other_styles)  # type: ignore

    return style_paths  # type: ignore


def apply_pyscopee_plot_style(other_styles: PlotStylePaths = None) -> None:
    """
    Applies the default ``pyscopee`` style to plots together with optional additional
    styles.

    Parameters
    ----------
    other_styles : :obj:`str`, :obj:`pathlib.Path`, [:obj:`str` or :obj:`pathlib.Path`], or ``None``, default=``None``
        The name of the style or a list of style names to apply in addition to the
        default style for plots in the ``pyscopee`` package.
        If ``None``, only the default style is applied.

    """  # noqa: E501

    mplstyle.use(_concatenate_pyscopee_with_other_styles(other_styles=other_styles))

    return


@contextmanager
def pyscopee_plot_style(
    other_styles: PlotStylePaths = None,
    enable: bool = True,
) -> Generator:
    """
    A context manager that temporarily applies the default ``pyscopee`` style to plots
    together with optional additional styles.

    Parameters
    ----------
    other_styles : :obj:`str`, [:obj:`str`], or ``None``, default=``None``
        The name of the style or a list of style names to apply in addition to the
        default style for plots in the ``pyscopee`` package.
        If ``None``, only the default style is applied.
    enable : :obj:`bool`, default=``True``
        Whether to enable the style within the context manager (``True``) or not
        (``False``).
        This is useful when it's not sure if the style should be enabled or not, and
        handling these two different cases would require two separate blocks of code
        that would be mostly identical.
        Please refer to the Notes section for details.

    Notes
    -----
    The contextmanager is enabled by default, but it can also be disabled by setting
    the `enable` parameter to ``False``.
    Let's say, there is a function that allows for enabling or disabling the style
    based on some condition, and the condition is only known at runtime.
    This could be written like this:

    ```python
    def some_function(with_style: bool) -> None:
        if with_style:
            with pyscopee_plot_style():
                fig, ax = plt.subplots()
                ax.plot([1, 2, 3], [4, 5, 6])
                # ... more code code could come here

        else:
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [4, 5, 6])
            # ... more code code could come here
    ```

    but this would be quite repetitive and if the ``else`` cannot be avoided by an
    early return, the code would become quite nested and less maintainable.
    Instead, the context manager can be used like this:

    ```python
    def some_function(with_style: bool) -> None:
        with pyscopee_plot_style(enable=with_style):
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [4, 5, 6])
            # ... more code code could come here
    ```

    """

    style_paths = _concatenate_pyscopee_with_other_styles(other_styles=other_styles)

    if enable:
        with mplstyle.context(style_paths):
            yield

    else:
        yield

    return
