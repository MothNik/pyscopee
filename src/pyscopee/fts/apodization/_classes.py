"""
Module :mod:`fts.apodization._classes`

This module provides class-interfaces to different apodization functions.

"""

# === Setup ===

__all__ = [
    "Boxcar",
    "CustomApodization",
    "Triangular",
    "ZeroMappedHyperbolicSine",
]

# === Imports ===

from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ..._utils import (
    Integer,
    RealNumeric,
    RealNumericArrayLike,
    apply_pyscopee_plot_style,
    get_validated_integer,
    get_validated_real_numeric,
    split_class_name_to_readable,
)
from ._functions import (
    WrappedApodizationFunction,
    as_apodization_function,
    boxcar,
    get_validated_xmax,
    not_implemented_apodization,
    triangular,
    zero_mapped_hyperbolic_sine,
)

# === Auxiliary Functions ===


def _get_validated_apodization_name(
    name: Optional[str],
    obj: object,
) -> str:
    """
    Validates the name of the apodization function if it is provided.
    If it is not provided, the class name is used after splitting it into a more
    readable format by inserting spaces between capital letters.

    Parameters
    ----------
    name : :class:`str` or ``None``
        The name of the apodization function.
    obj : :class:`object`
        The object whose class name is used if the name is not provided.

    Returns
    -------
    validated_name : :class:`str`
        The validated name of the apodization function.

    Raises
    ------
    TypeError
        If ``name`` is not of the expected type.

    """

    if isinstance(name, str):
        return name

    if name is None:
        return split_class_name_to_readable(obj)

    raise TypeError(
        f"Expected 'name' to be of type 'str' or 'None', but it is of type "
        f"{type(name)}."
    )


def _get_validated_evaluator(
    evaluator: Union[Callable, WrappedApodizationFunction, None],
) -> WrappedApodizationFunction:
    """
    Validates the evaluator function of the apodization function if it is provided.
    If it is not provided, a placeholder function is used that raises a
    :class:`NotImplementedError` when called.

    Parameters
    ----------
    evaluator : callable or ``None``
        The function that evaluates the apodization function at the given points.
        If it is not a validated apodization function, it is converted to one.
        If ``None``, the apodization function is set to a placeholder that raises a
        :class:`NotImplementedError` when called.

    Returns
    -------
    validated_evaluator : :class:`WrappedApodizationFunction`
        The validated evaluator function of the apodization function.

    Raises
    ------
    TypError or ValueError
        If the ``evaluator`` cannot be converted to a validated apodization function.

    """

    if evaluator is None:
        return not_implemented_apodization

    if hasattr(evaluator, "is_validated_apodization_function"):
        return evaluator

    return as_apodization_function(evaluator)


def _get_validated_exponent(
    exponent: RealNumeric,
) -> float:
    """
    Validates the ``exponent`` of the apodization function :class:`ZeroMappedHyperbolicSine`.

    Parameters
    ----------
    exponent : :class:`RealNumeric`
        The exponent to which the apodization function is raised.

    Returns
    -------
    validated_exponent : :class:`float`
        The validated exponent to which the apodization function is raised.

    Raises
    ------
    TypeError
        If ``exponent`` is not of the expected type.
    ValueError
        If ``exponent`` is not a real number ``>= 0``.

    """  # noqa: E501

    return get_validated_real_numeric(
        value=exponent,
        name="exponent",
        min_value=0.0,
        min_inclusive=True,
    )


# === Classes ===


class CustomApodization:
    """
    The base class for apodization functions that defines the basic methods like
    evaluation and visualisation for a uniform interface.

    Parameters
    ----------
    x_max : :class:`float` or :class:`int`, default=``1.0``
        The maximum value of the x-range over which the apodization function is defined.
        It must be a positive real number ``> 0``.
        With this, the x-range of ``[-1, 1]`` where apodization functions are typically
        defined is scaled to ``[-x_max, x_max]``.
    name : :class:`str` or ``None``, default=``None``
        The name of the apodization function.
        If ``None``, the name is set to the class name that was converted to a more
        readable form with spaces between the words.
    evaluator : callable or ``None``, default=``None``
        The function that evaluates the apodization function at the given points.
        Ideally, this function was decorated with the :func:`pyscopee.apodization.as_apodization_function`
        decorator that converts the function to an apodization function with verified
        signature and input validation. Please refer to the documentation of this
        decorator for more information on the required signature.
        If the evaluator is not a validated apodization function, an attempt is made to
        convert it to one.
        If ``None``, the apodization function is set to a placeholder that raises a
        :class:`NotImplementedError` when called.
    latex_representation : :class:`str` or ``None``, default=``None``
        The LaTeX representation of the apodization function that can be used in plots.
        No checks are made on the validity of the LaTeX representation.
        Please see ``latex_auto_complete`` and the Notes section for further details.
        If ``None``, there will be no LaTeX representation.
    latex_auto_complete : :class:`bool`, default=``True``
        Whether to automatically complete the LaTeX representation of the apodization
        function by framing it with a leading and trailing string and parsing the
        placeholders for the parameters (``True``) or not (``False``).
        Please refer to the Notes section for further details.

    Raises
    ------
    TypeError
        If ``x_max``, ``name``, or ``evaluator`` are not of the expected type.
    ValueError
        If ``x_max`` is not a positive real number ``> 0``.
    ValueError
        If ``evaluator`` cannot be converted to a validated apodization function.

    Notes
    -----
    If ``latex_auto_complete`` is ``False``, a custom LaTeX representation of the
    apodization function has to be provided in the ``latex_representation`` parameter.
    It has to be parsable by the :mod:`matplotlib` LaTeX engine as is (this is not
    checked). No additional modifications are made to the LaTeX representation.
    For example, the LaTeX representation

    ```
    f(x) = 1 - |x/x_max|
    ```

    for the :class:`Triangular` apodization function can be provided as

    ```
    r"f\\left(x\\right) = 1 - |\\frac{x}{x_{max}}|"
    ```

    For the case when ``latex_auto_complete`` is ``True``, the LaTeX representation
    of the apodization function is automatically completed by the class. The LaTeX
    representation of the apodization function is split into three parts like

    ```
    f(x) = some_function(x)     for -x_{max} <= x <= x_{max}
    ```

    Only the part ``some_function(x)`` has to be provided in the
    ``latex_representation`` while the rest is added automatically. The LaTeX
    representation is then framed by the strings ``r"$f\\left(x\\right) = "`` and
    ``r"\\text{     for }-x_{max}\\leq x\\leq x_{max}$"``. No ``"$"`` is required for
    ``some_function(x)``.
    Additionally, the LaTeX representation can include placeholders for the parameters
    of the apodization function. These placeholders are replaced by the actual values
    of the parameters when the LaTeX representation is accessed. The same applies to the
    parameter ``x_max``. All placeholders will be replaced by the actual values with a
    precision of three decimal places.
    For example, the :class:`Triangular` apodization function with ``x_max = 5.0`` can
    simply be represented as

    ```
    latex_representation = r"1-|\\frac{x}{x_{max}}|"
    ```

    which will be automatically completed to

    ```
    r"$f\\left(x\\right) = 1-|\\frac{x}{5.000}|\\text{     for }-5.000\\leq x\\leq 5.000$"
    ```

    when the LaTeX representation is accessed via the ``class.latex_representation``
    property.

    For an apodization function that has a parameter ``parameter`` with a value of
    ``2.5``, the example LaTeX representation

    ```
    latex_representation = r"\\exp(-\\frac{x^2}{2\\cdot parameter^2})"
    ```

    will be automatically completed to

    ```
    r"$f\\left(x\\right) = \\exp\\left(-\\frac{x^2}{2\\cdot 2.500^2}\\right)\\text{     for }-5.000\\leq x\\leq 5.000$"
    ```

    when the LaTeX representation is accessed via the ``class.latex_representation``
    property.

    """  # noqa: E501

    # --- Class attributes ---

    _latex_start: str = r"$f\left(x\right)\ =\ "
    _latex_stop: str = r"\text{     for }-x_{max}\leq x\leq x_{max}$"
    _call_params: Tuple[str, ...] = tuple()

    # --- Constructor ---

    def __init__(
        self,
        x_max: RealNumeric = 1.0,
        name: Optional[str] = None,
        evaluator: Union[Callable, WrappedApodizationFunction, None] = None,
        latex_representation: Optional[str] = None,
        latex_auto_complete: bool = True,
    ) -> None:

        self._x_max: float = get_validated_xmax(x_max=x_max)
        self._name: str = _get_validated_apodization_name(name=name, obj=self)
        self._evaluator: WrappedApodizationFunction = _get_validated_evaluator(
            evaluator=evaluator,
        )
        self._latex_representation: Optional[str] = latex_representation
        self._latex_auto_complete: bool = latex_auto_complete

        return

    # --- Properties ---

    @property
    def _call_kwargs(self) -> Dict[str, Any]:
        """
        The keyword arguments of the ``__call__`` method as a keyword argument
        dictionary.

        """

        call_kwargs = {key.lstrip("_"): getattr(self, key) for key in self._call_params}

        return call_kwargs

    @property
    def x_max(self) -> float:
        """
        The maximum value of the x-range over which the apodization function is defined.
        It is a positive real number ``> 0``.

        """

        return self._x_max

    @x_max.setter
    def x_max(self, value: RealNumeric) -> None:
        self._x_max = get_validated_xmax(x_max=value)
        return

    @property
    def name(self) -> str:
        """
        The name of the apodization function.

        """

        return self._name

    @name.setter
    def name(self, value: Optional[str]) -> None:
        self._name = _get_validated_apodization_name(name=value, obj=self)
        return

    @property
    def latex_representation(self) -> str:
        """
        The latex representation of the apodization function.
        It will be empty if no latex representation was provided at construction.

        """

        if self._latex_representation is None:
            return ""

        if not self._latex_auto_complete:
            return self._latex_representation

        latex_representation = self._latex_representation
        for key, value in self._call_kwargs.items():
            latex_representation = latex_representation.replace(key, f"{value:.3f}")

        return (self._latex_start + latex_representation + self._latex_stop).replace(
            "x_{max}", f"{self._x_max:.3f}"
        )

    # --- Magic Methods ---

    def __call__(self, x: RealNumericArrayLike) -> NDArray[np.float64]:
        """
        Evaluates the apodization function at the given points.

        Parameters
        ----------
        x : Array-like of shape (n,)
            The points at which to evaluate the apodization function.
            Negative entries are converted to positive ones under the assumption that
            the apodization function has even symmetry.
            Its length has to be at least 1.
            It is internally promoted to ``np.float64``.

        Returns
        -------
        apodization_values : :class:`numpy.ndarray` of shape (n,) of dtype ``np.float64``
            The values of the apodization function at the given points.

        Raises
        ------
        TypeError
            If ``x`` is not of the expected type.
        ValueError
            If ``x`` is an empty Array-like.
        ValueError
            If ``x``is not a real numeric 1D Array-like.

        """  # noqa: E501

        return self._evaluator(
            x=x,
            x_max=self._x_max,
            skip_validation=True,
            **self._call_kwargs,
        )

    # --- Methods ---

    def evaluate_with_fourier_transform(
        self,
        x_limit: RealNumeric = 5.0,
        x_num_points: Integer = 50_001,
        normalize: bool = True,
        real_fft: bool = True,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """
        Computes the apodization function and its Fourier transform.

        Parameters
        ----------
        x_limit : :class:`float` or :class:`int`, default=``5.0``
            The maximum value of the x-range over which the apodization function is
            computed before computing its Fourier transform. It is given as a multiple
            of ``x_max``, i.e., a value of ``5.0`` means that the apodization function
            is computed on the interval ``[-5 * x_max, 5 * x_max)`` with
            ``x_num_points`` equidistant points (the endpoint is excluded).
            It has to be a positive real number ``>= 1.0``.
            Increasing this number beyond ``1.0`` will result in a more densely sampled
            Fourier transform.
        x_num_points : :class:`int`, default=``50_001``
            The number of equidistant points to use for the computation of the
            Fourier transform. It will silently be rounded up to the next even number.
            It has to be a positive integer ``>= 100``.
            Increasing this number will allow for sampling higher frequencies in the
            Fourier transform.
        normalize : :class:`bool`, default=``True``
            Whether to normalize the apodization function to have a sum of 1 in the
            time/space domain (``True``) or not (``False``).
        real_fft : :class:`bool`, default=``True``
            Whether to use the real-valued Fast Fourier Transform (FFT) :func:`numpy.fft.rfft`
            (``True``) or the complex-valued FFT :func:`numpy.fft.fft` (``False``).

        Returns
        -------
        x : :class:`numpy.ndarray` of shape (x_num_points,) of dtype ``np.float64``
            The points at which the apodization function is computed.
            Its right endpoint is excluded.
        y : :class:`numpy.ndarray` of shape (x_num_points,) of dtype ``np.float64``
            The values of the apodization function at the given points.
            It is scaled to have a sum of 1 if ``normalize`` is ``True``.
        frequencies : :class:`numpy.ndarray` of shape (x_num_points,) or (x_num_points // 2 + 1,) of dtype ``np.float64``
            The frequencies at which the Fourier transform is computed.
            If ``real_fft`` is ``True``, only the positive frequencies are returned.
        fourier_transform : :class:`numpy.ndarray` of shape (x_num_points,) or (x_num_points // 2 + 1,) of dtype ``np.float64``
            The values of the Fourier transform at the given frequencies.
            If ``real_fft`` is ``True``, only the coefficients for the positive
            frequencies are returned.

        Raises
        ------
        TypeError
            If ``x_num_points`` is not of the expected type.
        ValueError
            If ``x_num_points`` is not a positive integer ``>= 100``.

        """  # noqa: E501

        x_num_points = get_validated_integer(
            value=x_num_points,
            name="x_num_points",
            min_value=100,
            min_inclusive=True,
        )

        # --- Computation ---

        # the number of points is silently rounded up to the next even number
        if x_num_points % 2 == 1:
            x_num_points += 1

        # the apodization function is evaluated on the interval
        # [-x_limit * x_max, x_limit * x_max], but the right endpoint is excluded
        x = np.linspace(
            start=-x_limit * self._x_max,
            stop=x_limit * self._x_max,
            num=x_num_points,
            endpoint=False,
            dtype=np.float64,
        )
        y = self(x=x)
        if normalize:
            y /= y.sum()

        # the Fourier transform is computed
        if real_fft:
            frequency_func = np.fft.rfftfreq
            fft_func = np.fft.rfft
        else:
            frequency_func = np.fft.fftfreq
            fft_func = np.fft.fft

        frequencies = frequency_func(
            n=x_num_points,
            d=(x[-1] - x[0]) / (x_num_points - 1),
        )
        # the point of x=0 is shifted to the beginning of the array to avoid imaginary
        # parts in the Fourier transform
        fourier_transform = fft_func(np.roll(y, shift=x_num_points // 2))

        return x, y, frequencies, fourier_transform.real

    # TODO: enable comparison of apodization functions
    def plot(
        self,
        fig: Optional[Figure] = None,
        x_compute_limit: RealNumeric = 5.0,
        x_num_points: Integer = 50_001,
        x_plot_limits: Optional[Tuple[RealNumeric, RealNumeric]] = None,
        frequency_plot_limits: Optional[Tuple[RealNumeric, RealNumeric]] = None,
        fourier_scale: Literal[
            "linear", "log", "logarithmic", "db", "decibel"
        ] = "linear",
        apodization_line_kwargs: Optional[Dict[str, Any]] = None,
        fourier_line_kwargs: Optional[Dict[str, Any]] = None,
        autoscale_y_axes: bool = True,
        fig_size: Tuple[float, float] = (12, 6),
        use_pyscopee_style: bool = True,
    ) -> Tuple[Figure, Axes, Axes]:
        """
        Plots the apodization function and its Fourier transform.

        Parameters
        ----------
        fig : :class:`matplotlib.figure.Figure` or ``None``, default=``None``
            The figure to use for the plot.
            If provided, it has to be a figure with two axes that can be accessed via
            ``fig.get_axes()[0]`` and ``fig.get_axes()[1]``.
            If ``None``, a new figure is created.
        x_compute_limit : :class:`float` or :class:`int`, default=5.0
            The maximum value of the x-range over which the apodization function is
            computed before computing its Fourier transform. It is given as a multiple
            of ``x_max``, i.e., a value of ``5.0`` means that the apodization function
            is computed on the interval ``[-5 * x_max, 5 * x_max]`` with
            ``x_num_points`` equidistant points.
            It has to be a positive real number ``>= 1.0``.
            Increasing this number beyond ``1.0`` will result in a more densely sampled
            Fourier transform.
        x_num_points : :class:`int`, default=50_001
            The number of equidistant points to use for the computation of the
            Fourier transform. It will silently be rounded up to the next even number.
            It has to be a positive integer ``>= 100``.
            Increasing this number will allow for sampling higher frequencies in the
            Fourier transform.
        x_plot_limits : (:class:`float` or :class:`int`, :class:`float` or :class:`int`) or ``None``, default=``None``
            The limits of the x-axis of the plot for the time/space domain.
            If the first element is larger than the second one, the plot will be
            flipped horizontally.
            If ``None``, the limits are set to ``(-1.05 * x_max, 1.05 * x_max)``.
        frequency_plot_limits : (:class:`float` or :class:`int`, :class:`float` or :class:`int`) or ``None``, default=``None``
            The limits of the x-axis of the plot for the frequency domain.
            If the first element is larger than the second one, the plot will be
            flipped horizontally.
            If ``None``, the limits are set to ``(-10.0 / x_max,  10.0 / x_max)``.
        fourier_scale : {``"linear"``, ``"log"``, ``"logarithmic``, ``"db"``, ``"decibel"``}, default=``"linear"``
            The scale of the y-axis of the Fourier transform plot.

            - ``"linear"`` will show the y-axis in linear scale.
            - ``"log"`` or ``"logarithmic"`` will show the y-axis in logarithmic scale for the absolute
                value.
            - ``"db"`` or ``"decibel"`` is similar to ``"log"``, but the y-axis is
                scaled in decibels.

            ``"log"``, ``"logarithmic"``, ``"db"``, and ``"decibel"`` require
            ``fourier_show_absolute`` to be ``True``.
        apodization_line_kwargs : :{:class:`str`: any}, default=``None``
            The keyword arguments for the line plot of the apodization function.
            If ``None``, the default line style is used, which is a red solid line
            with a width of 2.
        fourier_line_kwargs : :{:class:`str`: any}, default=``None``
            The keyword arguments for the line plot of the Fourier transform.
            If ``None``, the ``apodization_line_kwargs`` are used.
        autoscale_y_axes : :class:`bool`, default=``True``
            Whether to automatically scale the y-axes of the plots (``True``) or leave
            them as they are (``False``).
        fig_size : (:class:`float` or :class:`int`, :class:`float` or :class:`int`), default=(12, 6)
            The size of the figure in inches as a tuple of ``(width, height)``.
            It is only used if ``fig`` is ``None``.
        use_pyscopee_style : :class:`bool`, default=``True``
            Whether to use the default ``pyscopee`` style for the plot (``True``) or not
            (``False``).
            For other styles, the RC parameters of ``matplotlib`` have to be set
            manually either before calling this method or by using the
            Matplotlib style context manager.
            It is only applied if ``fig`` is ``None``.

        Returns
        -------
        fig : :class:`matplotlib.figure.Figure`
            The figure containing the plot.
        time_space_ax, frequency_ax : :class:`matplotlib.axes.Axes`
            The axes of the time/space domain and the frequency domain, respectively.

        Raises
        ------
        TypeError
            If ``x_compute_limit``, ``x_num_points``, ``x_plot_limits``,
            ``freq_plot_limits``, or ``fourier_scale`` are not of the expected type.
        ValueError
            If ``x_compute_limit`` is not a positive real number ``>= 1.0``.
        ValueError
            If ``x_num_points`` is not a positive integer ``>= 100``.
        ValueError
            If ``x_plot_limits`` or ``freq_plot_limits`` are not real numeric 2-tuples.
        ValueError
            If ``fourier_scale`` is not one of the allowed values.
        ValueError
            If ``fourier_scale`` is ``"log"``, ``"logarithmic"``, ``"db"``, or
            ``"decibel"`` and ``fourier_show_absolute`` is ``False``.
        ValueError
            If the provided figure does not have two axes.

        """  # noqa: E501

        # --- Nested Functions ---

        def get_validated_plot_limits(
            value: Optional[Tuple[RealNumeric, RealNumeric]],
            name: str,
            default_value: Tuple[float, float],
        ) -> Tuple[float, float]:
            """
            Validates the plot limits ``x_plot_limits`` or ``freq_plot_limits``.

            """

            if value is None:
                return default_value

            if not isinstance(value, tuple):
                raise TypeError(
                    f"Expected '{name}' to be a 2-tuple of real numeric values, "
                    f"but it is of type {type(value)}."
                )

            if len(value) != 2:
                raise ValueError(
                    f"Expected '{name}' to be a 2-tuple of real numeric values, "
                    f"but it is of length {len(value)}."
                )

            return tuple(  # type: ignore
                get_validated_real_numeric(
                    value=value,
                    name=f"{name}[{index}]",
                )
                for index, value in enumerate(value)
            )

        # --- Input Validation ---

        x_compute_limit = get_validated_real_numeric(
            value=x_compute_limit,
            name="x_compute_limit",
            min_value=1.0,
            min_inclusive=True,
        )

        x_num_points = get_validated_integer(
            value=x_num_points,
            name="x_num_points",
            min_value=100,
            min_inclusive=True,
        )

        x_plot_limits = get_validated_plot_limits(
            value=x_plot_limits,
            name="x_plot_limits",
            default_value=(-1.05 * self._x_max, 1.05 * self._x_max),
        )

        frequency_plot_limits = get_validated_plot_limits(
            value=frequency_plot_limits,
            name="freq_plot_limits",
            default_value=(-10.0 / self._x_max, 10.0 / self._x_max),
        )

        if not isinstance(fourier_scale, str):
            raise TypeError(
                f"Expected 'fourier_scale' to be of type str, but it is of type "
                f"{type(fourier_scale)}."
            )

        fourier_scale = fourier_scale.lower()  # type: ignore
        if fourier_scale not in {"linear", "log", "db", "decibel"}:
            raise ValueError(
                f"Expected 'fourier_scale' to be one of 'linear', 'log', or 'decibel', "
                f"but it is '{fourier_scale}'."
            )

        # --- Computation ---

        (
            x,
            y,
            frequencies,
            fourier_transform,
        ) = self.evaluate_with_fourier_transform(
            x_limit=x_compute_limit,
            x_num_points=x_num_points,
            normalize=True,
            real_fft=False,
        )

        if fourier_scale == "linear":
            fourier_transform = fourier_transform.real
            fourier_y_label = "Value Y(f)"
        elif fourier_scale == "log":
            fourier_transform = np.abs(fourier_transform.real)
            fourier_y_label = "Magnitude |Y(f)|"
        else:
            fourier_transform = np.abs(fourier_transform.real)
            with np.errstate(divide="ignore"):
                fourier_transform = 20.0 * np.log10(
                    fourier_transform / fourier_transform.max()
                )

            fourier_y_label = "Magnitude |Y(f)| in dB"

        # --- Plotting ---

        if use_pyscopee_style and fig is None:
            apply_pyscopee_plot_style()

        if fig is None:
            fig, axes = plt.subplots(ncols=2, figsize=fig_size)
        else:
            try:
                axes = np.array(fig.get_axes())
                if axes.ndim != 1 or axes.size != 2:
                    raise ValueError()

            except Exception as error:
                raise ValueError(
                    "Something is wrong with the provided figure. "
                    "It is expected to have two axes that can be accessed via "
                    "'fig.get_axes()[0]' and 'fig.get_axes()[1]'."
                ) from error

        # plot the apodization function
        if apodization_line_kwargs is None:
            apodization_line_kwargs = {"color": "red"}

        x_plot_indices = np.where((x >= x_plot_limits[0]) & (x <= x_plot_limits[1]))[0]
        axes[0].plot(
            x[x_plot_indices],
            y[x_plot_indices],
            **apodization_line_kwargs,
        )
        axes[0].set_xlabel("Time/Space x")
        axes[0].set_ylabel("Value y(x)")
        axes[0].set_xlim(*x_plot_limits)

        # plot the Fourier transform
        if fourier_line_kwargs is None:
            fourier_line_kwargs = apodization_line_kwargs

        frequencies = np.fft.fftshift(frequencies)
        fourier_transform = np.fft.fftshift(fourier_transform)
        freq_plot_indices = np.where(
            (frequencies >= frequency_plot_limits[0])
            & (frequencies <= frequency_plot_limits[1])
        )[0]

        axes[1].plot(
            frequencies[freq_plot_indices],
            fourier_transform[freq_plot_indices],
            **fourier_line_kwargs,
        )
        axes[1].set_xlabel("Frequency f")
        axes[1].yaxis.tick_right()
        axes[1].yaxis.set_label_position("right")
        axes[1].set_ylabel(fourier_y_label)
        axes[1].set_xlim(*frequency_plot_limits)

        if fourier_scale == "log":
            axes[1].set_yscale("log")

        fig.suptitle(f"{self._name}", fontsize=16, fontweight="bold", y=0.99)

        # the equation is written above the first plot, but only if it is available
        latex_representation = self.latex_representation
        if len(latex_representation) == 0:
            return fig, axes[0], axes[1]

        axes[0].text(
            x=0.5,
            y=0.92,
            s=self.latex_representation,
            horizontalalignment="center",
            verticalalignment="center",
            transform=fig.transFigure,
            fontsize=13,
        )

        return fig, axes[0], axes[1]


class Boxcar(CustomApodization):
    """
    The Boxcar apodization function which is defined as a constant function

    ```
    f(x) = 1
    ```

    within the interval ``[-x_max, x_max]``.

    Parameters
    ----------
    x_max : :class:`float` or :class:`int`, default=1.0
        The maximum value of the x-range over which the apodization function is defined.
        It must be a positive real number ``> 0``.
        With this, the x-range of ``[-1, 1]`` where apodization functions are typically
        defined is scaled to ``[-x_max, x_max]``.

    """

    # --- Constructor ---

    def __init__(self, x_max: RealNumeric = 1.0) -> None:
        super().__init__(
            x_max=x_max,
            name=None,
            evaluator=boxcar,
            latex_representation=r"1",
            latex_auto_complete=True,
        )

        return


class Triangular(CustomApodization):
    """
    The Triangular apodization function which is defined as a linear function

    ```
    f(x) = 1 - |x|
    ```

    within the interval ``[-x_max, x_max]``.

    Parameters
    ----------
    x_max : :class:`float` or :class:`int`, default=1.0
        The maximum value of the x-range over which the apodization function is defined.
        It must be a positive real number ``> 0``.
        With this, the x-range of ``[-1, 1]`` where apodization functions are typically
        defined is scaled to ``[-x_max, x_max]``.

    """

    # --- Constructor ---

    def __init__(self, x_max: RealNumeric = 1.0) -> None:
        super().__init__(
            x_max=x_max,
            name=None,
            evaluator=triangular,
            latex_representation=r"1-|\frac{x}{x_{max}}|",
            latex_auto_complete=True,
        )

        return


class ZeroMappedHyperbolicSine(CustomApodization):
    """
    The Zero-mapped hyperbolic sine apodization function which is defined as

    ```
    f(x) = (sinh(1 - (x / x_max)**2))**alpha / (sinh(1))**alpha``
    ```

    within the interval ``[-x_max, x_max]``.

    Parameters
    ----------
    x_max : :class:`float` or :class:`int`, default=1.0
        The maximum value of the x-range over which the apodization function is defined.
        It must be a positive real number ``> 0``.
        With this, the x-range of ``[-1, 1]`` where apodization functions are typically
        defined is scaled to ``[-x_max, x_max]``.
    exponent : :class:`float` or :class:`int`, default=5
        The exponent to which the apodization function is raised.
        It must be a real number ``>= 0``.
        ``0``will result in a :class:`Boxcar` apodization function.

    References
    ----------
    .. [1] Parker K. J., Apodization and Windowing Functions,
       Transactions on Ultrasonics, Ferroelectrics, and Frequency Control,
       Volume 60, Issue 6, 2013, pp. 1263 - 1271, DOI: 10.1109/TUFFC.2013.2691

    """

    # --- Class attributes ---

    _call_params: Tuple[str, ...] = ("_exponent",)

    # --- Constructor ---

    def __init__(
        self,
        x_max: RealNumeric = 1.0,
        exponent: RealNumeric = 5,
    ) -> None:
        super().__init__(
            x_max=x_max,
            name=None,
            latex_representation=(
                r"\frac{\left(\sinh\left("
                r"1-\left(\frac{x}{x_{max}}\right)^{2}\right)"
                r"\right)^{exponent}}"
                r"{\left(\sinh\left(1\right)\right)^{exponent}}"
            ),
            evaluator=zero_mapped_hyperbolic_sine,
            latex_auto_complete=True,
        )

        self._exponent: float = _get_validated_exponent(exponent=exponent)

        return

    # --- Properties ---

    @property
    def exponent(self) -> float:
        """
        The exponent to which the apodization function is raised.
        It is a real number ``>= 0``.

        """

        return self._exponent

    @exponent.setter
    def exponent(self, value: RealNumeric) -> None:
        self._exponent = _get_validated_exponent(exponent=value)
        return


# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# # apo = ZeroMappedHyperbolicSine(exponent=5)
# # apo.plot(
# #     fig=fig,
# #     x_num_points=500_001,
# #     fourier_scale="decibel",
# # )
# # # apo._plot_equation()
# apo = ZeroMappedHyperbolicSine(exponent=0.1)
# apo.plot(
#     x_num_points=500_001,
#     fourier_scale="decibel",
# )
# # # apo._plot_equation()
# print(boxcar(x=0.5, x_max=1.0))
# apo = Boxcar()
# apo.plot(
#     x_num_points=500_001,
#     fourier_scale="decibel",
# )
# # apo._plot_equation()
# apo = Triangular()
# apo.plot(
#     x_num_points=500_001,
#     fourier_scale="db",
# )

# apo = CustomApodization(
#     x_max=1.0,
#     name="Custom Apodization",
#     evaluator=boxcar,
#     latex_representation=r"$1$",
#     latex_auto_complete=False,
# )
# apo.plot(
#     x_num_points=500_001,
#     fourier_scale="linear",
#     apodization_line_kwargs={"color": "green"},
# )
# plt.show()
