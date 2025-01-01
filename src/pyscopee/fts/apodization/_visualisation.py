"""
Module :mod:`fts.apodization._visualisation`

This module provides functions for visualising apodization functions, particularly
comparing different apodization functions and their Fourier Transforms.

"""

# === Setup ===

__all__ = [
    "ApodizationWithFourierPlot",
    "FourierYScales",
    "PlotDataForApodizationWithFourier",
]

# === Imports ===

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text

from pyscopee._utils import RealNumeric

# === Models ===


class FourierYScales(str, Enum):
    """
    Enumeration of the possible y-scales for the Fourier Transform of apodization
    functions, namely,

    - :attr:`LINEAR`: linear scale
    - :attr:`LINEAR_ABSOLUTE`: linear scale with absolute values
    - :attr:`LOG`: logarithmic scale
    - :attr:`DECIBEL`: decibel scale


    """

    LINEAR = "linear"
    LINEAR_ABSOLUTE = "linear_absolute"
    LOGARITHMIC = "log"
    DECIBEL = "dB"

    @staticmethod
    def from_string(
        value: Literal[
            "linear",
            "lin",
            "linear_absolute",
            "lin_abs",
            "log",
            "logarithmic",
            "db",
            "decibel",
        ]
    ) -> "FourierYScales":
        """
        Converts a string to a :class:`FourierYScales` value.

        Parameters
        ----------
        value : {``"linear"``, ``"lin"``, ``linear_absolute``, ``"lin_abs"``, `"log"``, ``"logarithmic"``, ``"db"``, ``"decibel"``}
            The string to convert.

        Returns
        -------
        scale : :class:`FourierYScales`
            The corresponding :class:`FourierYScales` value.

        Raises
        ------
        TypeError
            If ``value`` is not a string.
        ValueError
            If ``value`` is not a valid :class:`FourierYScales` value.

        """  # noqa: E501

        if not isinstance(value, str):
            raise TypeError(
                f"Expected the Fourier y-scale value to be of type 'str', but got "
                f"'{type(value).__name__}'."
            )

        try:
            return {
                "linear": FourierYScales.LINEAR,
                "lin": FourierYScales.LINEAR,
                "linear_absolute": FourierYScales.LINEAR_ABSOLUTE,
                "lin_abs": FourierYScales.LINEAR_ABSOLUTE,
                "log": FourierYScales.LOGARITHMIC,
                "logarithmic": FourierYScales.LOGARITHMIC,
                "db": FourierYScales.DECIBEL,
                "decibel": FourierYScales.DECIBEL,
            }[value.lower()]

        except KeyError:
            allowed_values = ", ".join([f"'{scale.value}'" for scale in FourierYScales])
            raise ValueError(
                f"Got invalid Fourier y-scale '{value}'. "
                f"Allowed values are {allowed_values}."
            )


@dataclass
class PlotDataForApodizationWithFourier:
    """
    Represents the data for a plot of an apodization function and its Fourier Transform.

    """

    apodization_name: str
    latex_representation: Optional[str]
    apodization_x: np.ndarray
    apodization_y: np.ndarray
    apodization_line_kwargs: Dict[str, Any]

    fourier_x: np.ndarray
    fourier_y: np.ndarray
    fourier_line_kwargs: Dict[str, Any]


# === Auxiliary Functions ===


def _add_linebreaks_to_name(name: str, line_length: int = 20) -> str:
    """
    Adds line breaks to the name of an apodization function to make it more readable.

    Parameters
    ----------
    name : :class:`str`
        The name of the apodization function.
    line_length : :class:`int`, default=``20``
        The maximum length of each line.

    Returns
    -------
    formatted_name : :class:`str`
        The formatted name of the apodization function.

    """

    split_into_words = name.split(" ")
    formatted_name = ""
    line_length_counter = 0

    for word in split_into_words:
        if line_length_counter + len(word) + 1 > line_length:
            formatted_name += "\n"
            line_length_counter = 0

        formatted_name += f"{word} "
        line_length_counter += len(word) + 1

    return formatted_name.strip()


# === Classes ===


class ApodizationWithFourierPlot:
    """
    Represents a plot of an apodization function and its Fourier Transform.
    It handles the figure and the 2 axes objects for the plot.

    It can be used like a normal Matplotlib plot, by accessing the figure and the axes
    objects directly, e.g.,

    ```python
    apo_plot.figure  # or apo_plot.fig
    apo_plot.apodization_axes  # or apo_plot.axes[0]
    apo_plot.fourier_axes  # or apo_plot.axes[1]
    ```

    Parameters
    ----------
    figure : :class:`matplotlib.figure.Figure`
        The figure object for the plot.
    apodization_axes, fourier_axes : :class:`matplotlib.axes.Axes`
        The axes objects for the apodization and Fourier Transform plots, respectively.
    apodization_x_limits, apodization_y_limits : (:class:`float`, :class:`float`) or ``None``, default=``None``
        The x- and y-axis limits for the apodization plot.
        If ``None``, the limits are given by the plotted data.
    fourier_x_limits, fourier_y_limits : (:class:`float`, :class:`float`) or ``None``, default=``None``
        The x- and y-axis limits for the Fourier Transform plot.
        If ``None``, the limits are given by the plotted data.
    fourier_y_scale : :class:`FourierYScales`, default=``FourierYScales.LINEAR``
        The scale of the y-axis for the Fourier Transform plot.

    Notes
    -----
    For convenience, the plot can be created with the :meth:`create` method, which
    creates the figure and axes objects automatically.

    """  # noqa: E501

    # --- Class Attributes ---

    class _AxesKinds(str, Enum):
        """
        Enumeration of the kinds of axes in the plot.

        """

        APODIZATION = "apodization"
        FOURIER = "fourier"

    # the standard padding multiplier for the y-axis limits
    _Y_AXIS_PADDING = 0.05
    # the padding multiplier for logarithmic y-axis limits
    _LOG_Y_AXIS_PADDING = 2.5

    # --- Constructor ---

    def __init__(
        self,
        figure: Figure,
        apodization_axes: Axes,
        fourier_axes: Axes,
        apodization_x_limits: Optional[Tuple[float, float]] = None,
        apodization_y_limits: Optional[Tuple[float, float]] = None,
        fourier_x_limits: Optional[Tuple[float, float]] = None,
        fourier_y_limits: Optional[Tuple[float, float]] = None,
        fourier_y_scale: FourierYScales = FourierYScales.LINEAR,
    ):

        # the canvas and axes objects
        self.figure: Figure = figure
        self.apodization_axes: Axes = apodization_axes
        self.fourier_axes: Axes = fourier_axes

        # the axes limits and scales
        self.apodization_x_limits: Optional[Tuple[float, float]] = apodization_x_limits
        self.apodization_y_limits: Optional[Tuple[float, float]] = apodization_y_limits
        self.fourier_x_limits: Optional[Tuple[float, float]] = fourier_x_limits
        self.fourier_y_limits: Optional[Tuple[float, float]] = fourier_y_limits
        self.fourier_y_scale: FourierYScales = fourier_y_scale

        # special elements within the plot
        self.figure_title_textbox: Optional[Text] = None
        self.latex_representation_textbox: Optional[Text] = None

        # auxiliary attributes
        self.has_data: bool = False

    # --- Properties ---

    @property
    def fig(self) -> Figure:
        """
        The figure object of the plot.

        """

        return self.figure

    @property
    def axes(self) -> Tuple[Axes, Axes]:
        """
        The axes objects of the plot.

        """

        return self.apodization_axes, self.fourier_axes

    # --- Internal Methods ---

    def _plot_data(
        self,
        data: PlotDataForApodizationWithFourier,
    ) -> None:
        """
        Updates the plot with the new dataset.

        Parameters
        ----------
        data : :class:`PlotDataForApodizationWithFourier`
            The new dataset to add to the plot.

        """

        # for the label, the name of the apodization function is formatted to make it
        # more readable
        label = _add_linebreaks_to_name(
            name=data.apodization_name,
            line_length=20,
        )

        # the data is plotted
        for axes, which_axes in zip(
            (self.apodization_axes, self.fourier_axes),
            (self._AxesKinds.APODIZATION, self._AxesKinds.FOURIER),
        ):
            x_values = getattr(data, f"{which_axes}_x")
            y_values = getattr(data, f"{which_axes}_y")
            line_kwargs = getattr(data, f"{which_axes}_line_kwargs")

            axes.plot(x_values, y_values, label=label, **line_kwargs)
            label = None  # to avoid adding the same label multiple times

        return

    def _clear_previous_data(self) -> None:
        """
        Clears the previous data from the plot.

        """

        self.apodization_axes.cla()
        self.fourier_axes.cla()
        self.figure_title_textbox = None
        self.latex_representation_textbox = None

        return

    def _update_axis_limits(self, data: PlotDataForApodizationWithFourier) -> None:
        """
        Updates the limits of the axes in the plot.

        Parameters
        ----------
        data : :class:`PlotDataForApodizationWithFourier`
            The new dataset to add to the plot.

        """

        for ax, which_axes in zip(
            (self.apodization_axes, self.fourier_axes),
            (self._AxesKinds.APODIZATION, self._AxesKinds.FOURIER),
        ):

            # first, the x-axis limits are set
            x_limits = getattr(self, f"{which_axes}_x_limits")
            # if the x-axis limits are set, this needs to be kept in mind when
            # specifying the y-axis limits
            x_limits_specified = x_limits is not None
            if x_limits_specified:
                ax.set_xlim(x_limits)

            # then, the y-axis limits are set
            y_limits = getattr(self, f"{which_axes}_y_limits")
            # if the y-axis limits are set, they are just used as they are
            if y_limits is not None:
                ax.set_ylim(y_limits)
                continue

            # if there were no x-axis and no y-axis limits specified, the limits are
            # set by the data
            if not x_limits_specified:
                continue

            # if the y-axis limits are not set, but the x-axis limits are, the y-axis
            # limits are adjusted to the data within the x-axis limits
            x_values = getattr(data, f"{which_axes}_x")
            y_values = getattr(data, f"{which_axes}_y")
            index_from = np.searchsorted(x_values, x_limits[0], side="left")
            index_to = np.searchsorted(x_values, x_limits[1], side="right")

            if index_from == index_to:
                continue

            y_values_in_limits = y_values[index_from:index_to]
            y_min, y_max = y_values_in_limits.min(), y_values_in_limits.max()

            # some padding is added to the y-axis limits, but this depends on the scale
            # for any other scaling than logarithmic, the padding is a fixed percentage
            # of the range
            if (
                which_axes == self._AxesKinds.APODIZATION
                or self.fourier_y_scale != FourierYScales.LOGARITHMIC
            ):
                y_padding = self._Y_AXIS_PADDING * (y_max - y_min)
                y_limits = (y_min - y_padding, y_max + y_padding)

            # for logarithmic scaling, the limits need to be set as a multiple of the
            # minimum and maximum values to avoid distortion by negative values
            else:
                y_limits = (
                    y_min / self._LOG_Y_AXIS_PADDING,
                    y_max * self._LOG_Y_AXIS_PADDING,
                )

            ax.set_ylim(*y_limits)
            self.__setattr__(f"{which_axes}_y_limits", y_limits)

        return

    def _update_plot_labeling(
        self,
        data: PlotDataForApodizationWithFourier,
    ) -> None:
        """
        Updates the labeling of the plot with the new dataset.

        Parameters
        ----------
        data : :class:`PlotDataForApodizationWithFourier`
            The new dataset to add to the plot.

        """

        # otherwise, the plot is set up for the first dataset with all the labels
        # and titles
        # 1) the time-space domain axes
        self.apodization_axes.set_xlabel("Time/Space x")
        self.apodization_axes.set_ylabel("Value y(x)")

        # 2) the frequency domain axes
        self.fourier_axes.set_xlabel("Frequency f")
        self.fourier_axes.yaxis.tick_right()
        self.fourier_axes.yaxis.set_label_position("right")

        if self.fourier_y_scale == FourierYScales.LOGARITHMIC:
            self.fourier_axes.set_yscale("log")

        fourier_label = {
            FourierYScales.LINEAR: "Value Y(f)",
            FourierYScales.LINEAR_ABSOLUTE: "Magnitude |Y(f)|",
            FourierYScales.LOGARITHMIC: "Magnitude |Y(f)|",
            FourierYScales.DECIBEL: "Magnitude |Y(f)| in dB",
        }[self.fourier_y_scale]

        self.fourier_axes.set_ylabel(fourier_label)

        # 3) the LaTeX representation textbox
        if self.latex_representation_textbox is not None:
            raise AssertionError(
                "The LaTeX representation textbox should not be set for the first "
                "dataset."
            )

        if data.latex_representation is not None:
            self.latex_representation_textbox = self.apodization_axes.text(
                x=0.5,
                y=0.92,
                s=data.latex_representation,
                horizontalalignment="center",
                verticalalignment="center",
                transform=self.figure.transFigure,
                fontsize=13,
            )

        # 4) the title of the figure
        self.figure_title_textbox = self.figure.suptitle(
            f"{data.apodization_name}",
            fontsize=16,
            fontweight="bold",
            y=0.99,
        )

        return

    def plot(self, data: PlotDataForApodizationWithFourier) -> None:
        """
        Adds a dataset for an apodization function and its Fourier Transform to the
        plot.
        If data were already added to the plot, the plot is cleared before adding the
        new data.

        Parameters
        ----------
        data : :class:`PlotDataForApodizationWithFourier`
            The other dataset to add to the plot.
            It will be deleted after adding it to the plot.

        """

        # all the data is added to the plot
        self._plot_data(data=data)
        self._update_axis_limits(data=data)

        # the plot is updated
        self._update_plot_labeling(data=data)

        # the figure is redrawn
        self.figure.canvas.draw()

        return

    # --- Methods ---

    @staticmethod
    def create(
        figsize: Tuple[RealNumeric, RealNumeric] = (12, 8),
        apodization_x_limits: Optional[Tuple[float, float]] = None,
        apodization_y_limits: Optional[Tuple[float, float]] = None,
        fourier_x_limits: Optional[Tuple[float, float]] = None,
        fourier_y_limits: Optional[Tuple[float, float]] = None,
        fourier_y_scale: FourierYScales = FourierYScales.LINEAR,
    ) -> "ApodizationWithFourierPlot":
        """
        Creates a new :class:`ApodizationWithFourierPlot` instance without the need to
        create the figure and axes objects manually.

        fig_size : (:class:`float` or :class:`int`, :class:`float` or :class:`int`), default=``(12, 8)``
            The size of the figure in inches as a tuple of ``(width, height)``.
        apodization_x_limits, apodization_y_limits : (:class:`float`, :class:`float`) or ``None``, default=``None``
            The x- and y-axis limits for the apodization plot.
            If ``None``, the limits are given by the plotted data.
        fourier_x_limits, fourier_y_limits : (:class:`float`, :class:`float`) or ``None``, default=``None``
            The x- and y-axis limits for the Fourier Transform plot.
            If ``None``, the limits are given by the plotted data.
        fourier_y_scale : :class:`FourierYScales`, default=``FourierYScales.LINEAR``
            The scale of the y-axis for the Fourier Transform plot.

        Returns
        -------
        plot : :class:`ApodizationWithFourierPlot`
            The new plot instance.

        """  # noqa: E501

        # the figure and the axes are created
        fig, (apodization_axes, fourier_axes) = plt.subplots(
            ncols=2,
            figsize=figsize,
        )

        return ApodizationWithFourierPlot(
            figure=fig,
            apodization_axes=apodization_axes,
            fourier_axes=fourier_axes,
            apodization_x_limits=apodization_x_limits,
            apodization_y_limits=apodization_y_limits,
            fourier_x_limits=fourier_x_limits,
            fourier_y_limits=fourier_y_limits,
            fourier_y_scale=fourier_y_scale,
        )
