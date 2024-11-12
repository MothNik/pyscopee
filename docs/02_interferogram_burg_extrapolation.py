"""
This example script shows how to extrapolate interferograms beyond their original range
using the Burg algorithm implemented in ``pyscopee``.

"""

# === Imports ===

import os

import numpy as np
from matplotlib import pyplot as plt

from pyscopee import augment, black_body_spectrum
from pyscopee.spectra_simulate import (
    atmospheric_transmittance,
    window_three_segment_smooth_cutoff,
)

plt.style.use(os.path.join(os.path.dirname(__file__), "./pyscopee.mplstyle"))

# === Constants ===

# the specifications of the blackbody radiation
WAVENUMBERS_FROM = 0.0  # 1 / cm
WAVENUMBERS_TO = 100_000.0  # 1 / cm
NUM_WAVENUMBERS = 1_000_001
BLACK_BODY_TEMPERATURE = 1_300.0  # K

# the specification of the detector cutoff for the window
WINDOW_RAMP_UP_BOUNDS = (400.0, 1_500.0)
WINDOW_RAMP_UP_EXPONENT = 2
WINDOW_RAMP_DOWN_BOUNDS = (2_500.0, 6_000.0)
WINDOW_RAMP_DOWN_EXPONENT = 1.5

# the wavenumber resolution of the truncated interferogram
WAVENUMBER_RESOLUTION_TRUNCATED = 4.0  # 1 / cm

# the optical path difference fit regions for the Burg autoregression
BURG_FIT_REGIONS = [
    (-0.125, -0.1),  # cm
    (0.1, 0.125),  # cm
]
# the order of the Burg autoregression
BURG_ORDER = 2_500
# the Tikhonov regularization parameter for the Burg autoregression
BURG_REGULARIZATION = 1e-10
# the number of extrapolated points on each side of the interferogram
NUM_EXTRAPOLATED_POINTS = 20_000

# the path where to save the resulting plot
PLOT_FILEPATH = "./example_plots/02_interferogram_burg_extrapolate.png"

assert NUM_WAVENUMBERS % 2 == 1, "NUM_WAVENUMBERS must be odd"

# === Main ===

# --- Spectrum Generation ---

# the wavenumbers are set up
wavenumbers = np.linspace(
    start=WAVENUMBERS_FROM,
    stop=WAVENUMBERS_TO,
    num=NUM_WAVENUMBERS,
)
delta_wavenumbers = (WAVENUMBERS_TO - WAVENUMBERS_FROM) / (NUM_WAVENUMBERS - 1)

# the blackbody radiation spectrum is computed ...
spectrum = black_body_spectrum(
    wavenumbers=wavenumbers,
    temperature=BLACK_BODY_TEMPERATURE,
    temperature_unit="K",
)
# ... together with the window function that represents the detector cutoff
window = window_three_segment_smooth_cutoff(
    x=wavenumbers,
    x_min1=WINDOW_RAMP_UP_BOUNDS[0],
    x_max1=WINDOW_RAMP_UP_BOUNDS[1],
    x_min2=WINDOW_RAMP_DOWN_BOUNDS[0],
    x_max2=WINDOW_RAMP_DOWN_BOUNDS[1],
    exponent1=WINDOW_RAMP_UP_EXPONENT,
    exponent2=WINDOW_RAMP_DOWN_EXPONENT,
)

# the blackbody radiation spectrum is multiplied with the window function to mimic the
# detector cutoff
spectrum_with_cutoff = spectrum * window
indices_for_atmosphere = np.logical_and(
    wavenumbers >= WINDOW_RAMP_UP_BOUNDS[0],
    wavenumbers <= WINDOW_RAMP_DOWN_BOUNDS[1],
)

# on top of the cutoff, the atmospheric transmittance in the light path is included
# NOTE: this requires ``radis`` to be installed
# NOTE: this API will likely change in the future
atmosphere_transmittance = np.zeros_like(wavenumbers)
atmosphere_transmittance[indices_for_atmosphere] = atmospheric_transmittance(
    wavenumbers=wavenumbers[indices_for_atmosphere],
)
spectrum_with_atmosphere = spectrum_with_cutoff * atmosphere_transmittance

# --- Interferogram Generation ---

# the interferogram is computed by taking the inverse Fourier transform
interferogram_size = 2 * NUM_WAVENUMBERS - 1
interferogram = np.fft.irfft(spectrum_with_atmosphere, n=interferogram_size)
# NOTE: the interferogram is has to be re-arranged to have the zero wavenumber in the
#       center
interferogram = np.roll(interferogram, shift=NUM_WAVENUMBERS - 1)
optical_path_difference = np.linspace(
    start=-(0.5 / delta_wavenumbers),  # Nyquist criterion
    stop=(0.5 / delta_wavenumbers),
    num=interferogram_size,
)

# --- Interferogram Truncation and Extrapolation ---

# the interferogram is truncated to the specified wavenumber resolution
truncated_max_opd = 1 / (2 * WAVENUMBER_RESOLUTION_TRUNCATED)
truncation_keep_indices_from = np.searchsorted(
    optical_path_difference,
    -truncated_max_opd,
    side="left",
)
truncation_keep_indices_to = np.searchsorted(
    optical_path_difference,
    truncated_max_opd,
    side="right",
)
truncated_interferogram = interferogram.copy()
truncated_interferogram[:truncation_keep_indices_from] = 0.0
truncated_interferogram[truncation_keep_indices_to:] = 0.0

# the spectrum of the truncated interferogram is computed by taking the Fourier
# transform after moving the centerburst back to the zero optical path difference
truncated_spectrum = np.fft.rfft(
    np.roll(truncated_interferogram, shift=-(NUM_WAVENUMBERS - 1)),
)

# the interferogram is extrapolated using the Burg algorithm
fit_basis = []
for fit_region in BURG_FIT_REGIONS:
    fit_indices = np.where(
        np.logical_and(
            optical_path_difference >= fit_region[0],
            optical_path_difference <= fit_region[1],
        )
    )[0]
    fit_basis.append(truncated_interferogram[fit_indices])

ar_coeffs = augment.arburg(
    xs=fit_basis,
    order=BURG_ORDER,
    tikhonov_lambda=BURG_REGULARIZATION,
    jit=True,
)

extrapolation_left_side = augment.extrapolate_autoregressive(
    x=truncated_interferogram[
        truncation_keep_indices_from : truncation_keep_indices_from + BURG_ORDER
    ],
    ar_coeffs=ar_coeffs,
    pad_width=(NUM_EXTRAPOLATED_POINTS, 0),
    jit=True,
)[0:NUM_EXTRAPOLATED_POINTS]

extrapolation_right_side = augment.extrapolate_autoregressive(
    x=truncated_interferogram[
        truncation_keep_indices_to - BURG_ORDER : truncation_keep_indices_to
    ],
    ar_coeffs=ar_coeffs,
    pad_width=(0, NUM_EXTRAPOLATED_POINTS),
    jit=True,
)[-NUM_EXTRAPOLATED_POINTS:]

augmented_interferogram = truncated_interferogram.copy()
augmented_interferogram[
    truncation_keep_indices_from
    - NUM_EXTRAPOLATED_POINTS : truncation_keep_indices_from
] = extrapolation_left_side

augmented_interferogram[
    truncation_keep_indices_to : truncation_keep_indices_to + NUM_EXTRAPOLATED_POINTS
] = extrapolation_right_side

# the spectrum of the extrapolated interferogram is computed by taking the Fourier
# transform after moving the centerburst back to the zero optical path difference
extrapolated_spectrum = np.fft.rfft(
    np.roll(augmented_interferogram, shift=-(NUM_WAVENUMBERS - 1)),
)

# --- In

# the results are plotted

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
ax.plot(
    wavenumbers,
    truncated_spectrum,
    label="Truncated Spectrum",
    color="orange",
)
ax.plot(
    wavenumbers,
    extrapolated_spectrum,
    label="Extrapolated Spectrum",
    color="cyan",
)

fig_interferogram, ax_interferogram = plt.subplots(
    figsize=(12, 8),
)

ax_interferogram.plot(
    optical_path_difference,
    interferogram,
    label="Interferogram",
    color="red",
    linewidth=3,
)
ax_interferogram.plot(
    optical_path_difference,
    truncated_interferogram,
    label="Truncated Interferogram",
    color="blue",
    linewidth=2,
)
ax_interferogram.plot(
    optical_path_difference,
    augmented_interferogram,
    label="Extrapolated Interferogram",
    color="cyan",
    linewidth=1,
)

ax_interferogram.set_xlabel(r"Optical Path Difference $\left(cm\right)$")
ax_interferogram.set_ylabel(r"Intensity")

plt.show()
