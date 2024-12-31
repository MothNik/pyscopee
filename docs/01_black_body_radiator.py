"""
This example script shows the blackbody radiation function implemented in ``pyscopee``.

"""

# === Imports ===

import os

import numpy as np
from matplotlib import pyplot as plt

from pyscopee import apply_pyscopee_plot_style, black_body_peak, black_body_spectrum

apply_pyscopee_plot_style()

# === Constants ===

PLOT_FILEPATH = "./example_plots/01_black_body_radiator.png"

# === Main ===

# the wavenumbers and temperatures are set up
wavenumbers = np.linspace(
    start=0.0,
    stop=15_000.0,
    num=5_000,
)
temperature_start = 100.0
temperature_step = 100.0
num_temperature_steps = 20

temperatures = (
    temperature_start
    + np.arange(
        start=0,
        stop=num_temperature_steps,
        step=1,
        dtype=np.int64,
    )
    * temperature_step
)
black_body_peaks_specs = []

# the blackbody radiation spectra are computed one by one and plotted
fig, ax = plt.subplots(
    figsize=(12, 8),
)

colors = plt.cm.copper(  # type: ignore
    np.linspace(
        start=0.0,
        stop=1.0,
        num=num_temperature_steps,
    )
)

for temperature, color in zip(temperatures, colors):
    spectrum = black_body_spectrum(
        wavenumbers=wavenumbers,
        temperature=temperature,
        temperature_unit="K",
    )
    black_body_peaks_specs.append(black_body_peak(temperature=temperature))

    ax.plot(
        wavenumbers,
        spectrum,
        color=color,
    )

ax.plot(
    [peak[0] for peak in black_body_peaks_specs],
    [peak[1] for peak in black_body_peaks_specs],
    color="black",
    marker="o",
    linestyle="--",
    label="Peak",
)

# a colorbar is added
sm = plt.cm.ScalarMappable(
    cmap=plt.cm.copper,  # type: ignore
    norm=plt.Normalize(  # type: ignore
        vmin=temperatures.min(),
        vmax=temperatures.max(),
    ),
)
sm.set_array([])
fig.colorbar(sm, ax=ax, label="Temperature (K)")

# the plot is finalised
ax.legend()
ax.set_xlabel(r"Wavenumbers $\tilde{\nu}\ \left(cm^{-1}\right)$")
ax.set_ylabel(
    r"Blackbody Radiation Spectrum $\left(\frac{W \cdot cm}{m^2 \cdot sr}\right)$"
)

ax.set_xlim(wavenumbers.min(), wavenumbers.max())

# the plot is saved ...
if os.getenv("pyscopee_DEVELOPER", "false").lower() == "true":
    plt.savefig(os.path.join(os.path.dirname(__file__), PLOT_FILEPATH))

# ... and shown
plt.show()
