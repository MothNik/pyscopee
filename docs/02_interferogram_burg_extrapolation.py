"""
This example script shows how to extrapolate interferograms beyond their original range
using the Burg algorithm implemented in ``pyscopee``.

"""

# === Imports ===

import os

import numpy as np
from matplotlib import pyplot as plt

from pyscopee import black_body_spectrum
from pyscopee.spectra_simulate import (
    atmospheric_transmittance,
    window_three_segment_smooth_cutoff,
)

plt.style.use(os.path.join(os.path.dirname(__file__), "./pyscopee.mplstyle"))

# === Constants ===

# the specifications of the blackbody radiation
WAVENUMBERS_FROM = 0.0  # 1 / cm
WAVENUMBERS_TO = 100_000.0  # 1 / cm
NUM_WAVENUMBERS = 1_000_000
BLACK_BODY_TEMPERATURE = 1_300.0  # K

# the specification of the detector cutoff for the window
WINDOW_RAMP_UP_BOUNDS = (400.0, 1_500.0)
WINDOW_RAMP_UP_EXPONENT = 2
WINDOW_RAMP_DOWN_BOUNDS = (2_500.0, 6_000.0)
WINDOW_RAMP_DOWN_EXPONENT = 1.5


# the path where to save the resulting plot
PLOT_FILEPATH = "./example_plots/02_interferogram_burg_extrapolate.png"

# === Main ===

# the wavenumbers are set up
wavenumbers = np.linspace(
    start=WAVENUMBERS_FROM,
    stop=WAVENUMBERS_TO,
    num=NUM_WAVENUMBERS,
)

# the blackbody radiation spectrum is computed ...
spectrum = black_body_spectrum(
    wavenumbers=wavenumbers,
    temperature=BLACK_BODY_TEMPERATURE,
    temperature_unit="K",
)
# ... together with the window function
window = window_three_segment_smooth_cutoff(
    x=wavenumbers,
    x_min1=WINDOW_RAMP_UP_BOUNDS[0],
    x_max1=WINDOW_RAMP_UP_BOUNDS[1],
    x_min2=WINDOW_RAMP_DOWN_BOUNDS[0],
    x_max2=WINDOW_RAMP_DOWN_BOUNDS[1],
    exponent1=WINDOW_RAMP_UP_EXPONENT,
    exponent2=WINDOW_RAMP_DOWN_EXPONENT,
)

# the blackbody radiation spectrum is multiplied with the window function
spectrum_with_cutoff = spectrum * window
indices_for_atmosphere = np.logical_and(
    wavenumbers >= WINDOW_RAMP_UP_BOUNDS[0],
    wavenumbers <= WINDOW_RAMP_DOWN_BOUNDS[1],
)
atmosphere_transmittance = np.zeros_like(wavenumbers)
atmosphere_transmittance[indices_for_atmosphere] = atmospheric_transmittance(
    wavenumbers=wavenumbers[indices_for_atmosphere],
)
spectrum_with_atmosphere = spectrum_with_cutoff * atmosphere_transmittance

# a tiny baseline is added
spectrum_with_atmosphere += 1e-3 * np.max(spectrum_with_atmosphere)

# the interferogram


fig, ax = plt.subplots(
    figsize=(12, 8),
)

ax.plot(
    wavenumbers,
    spectrum,
    label="Blackbody Radiation Spectrum",
    color="blue",
    lw=2,
)

ax.plot(
    wavenumbers,
    window * np.max(spectrum),
    label="Window Function",
    color="green",
    lw=2,
)

ax.plot(
    wavenumbers,
    spectrum_with_cutoff,
    label="Spectrum with Window Function",
    color="red",
    lw=2,
)
ax.plot(
    wavenumbers,
    spectrum_with_atmosphere,
    label="Spectrum with Atmosphere",
    color="purple",
)

fig2, ax2 = plt.subplots(
    figsize=(12, 8),
)

ax2.plot(
    np.fft.irfft(spectrum_with_atmosphere),
    label="Interferogram",
    color="red",
)

plt.show()
