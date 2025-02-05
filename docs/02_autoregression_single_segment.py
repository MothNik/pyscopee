"""
This example script shows the :func:`arburg` function implemented in ``pyscopee``
when applied to a single signal.

"""

# === Imports ===

import os
from time import perf_counter_ns

import numpy as np
from matplotlib import pyplot as plt

from pyscopee import apply_pyscopee_plot_style
from pyscopee.augment import (
    ar_ordinary_least_squares,
    arburg,
    extrapolate_autoregressive,
)

apply_pyscopee_plot_style()

# === Constants ===

# the size of the signal
SIGNAL_SIZE = 1_000
# the order of the autoregressive model
AR_ORDER = 800
# the number of samples to extrapolate
NUM_SAMPLES_EXTRAPOLATED = 1_000
# the noise level
NOISE_LEVEL = 0.05

# the path to save the plot
PLOT_FILEPATH = "./example_plots/02_autoregression_single_segment.png"


# === Main ===

# the signal is set up based upon the original C++ code
# vector<double> original( 128, 0.0 );
# for ( size_t i = 0; i < original.size(); i++ )
#  {
#  original[ i ] = cos( i * 0.01 ) + 0.75 *cos( i * 0.03 )
#  + 0.5 *cos( i * 0.05 ) + 0.25 *cos( i * 0.11 );
#  }
#  // GET LINEAR PREDICTION COEFFICIENTS
#  vector<double> coeffs( 4, 0.0 )

np.random.seed(0)
x = np.arange(
    start=-NUM_SAMPLES_EXTRAPOLATED,
    stop=SIGNAL_SIZE + NUM_SAMPLES_EXTRAPOLATED,
    step=1,
    dtype=np.float64,
)
y = (
    np.cos(x * 0.01)
    + 0.75 * np.cos(x * 0.03)
    + 0.5 * np.cos(x * 0.05)
    + 0.25 * np.cos(x * 0.11)
)
y_noisy = y + NOISE_LEVEL * np.random.randn(y.size)

# --- Burg's method ---

# the AR coefficients are computed using the Burg method
x_truncated = x[NUM_SAMPLES_EXTRAPOLATED:-NUM_SAMPLES_EXTRAPOLATED]
y_truncated = y_noisy[NUM_SAMPLES_EXTRAPOLATED:-NUM_SAMPLES_EXTRAPOLATED]

start_time = perf_counter_ns()
ar_coeffs = arburg(
    xs=y_truncated,
    order=AR_ORDER,
    tikhonov_lambda=1e1,
)
print(
    f"Burg AR coefficients computation took "
    f"{(1e-6 * (perf_counter_ns() - start_time)):.3f} milliseconds."
)

# the signal is extrapolated with the Burg AR coefficients
start_time = perf_counter_ns()
y_extrapolated_burg = extrapolate_autoregressive(
    x=y_truncated,
    ar_coeffs=ar_coeffs,
    pad_width=(NUM_SAMPLES_EXTRAPOLATED, NUM_SAMPLES_EXTRAPOLATED),
)
print(
    f"Burg Extrapolation took {(1e-6 * (perf_counter_ns() - start_time)):.3f} "
    f"milliseconds."
)

# --- Ordinary least squares ---

# the AR coefficients are computed using the ordinary least squares method
start_time = perf_counter_ns()
ar_coeffs = ar_ordinary_least_squares(
    xs=y_truncated,
    order=AR_ORDER,
    tikhonov_lambda=1e1,
    lstsq_solver="symmetric",
)
print(
    f"OLS AR coefficients computation took "
    f"{(1e-6 * (perf_counter_ns() - start_time)):.3f} milliseconds."
)

# the signal is extrapolated with the OLS AR coefficients
start_time = perf_counter_ns()
y_extrapolated_ols = extrapolate_autoregressive(
    x=y_truncated,
    ar_coeffs=ar_coeffs,
    pad_width=(NUM_SAMPLES_EXTRAPOLATED, NUM_SAMPLES_EXTRAPOLATED),
)
print(
    f"OLS Extrapolation took {(1e-6 * (perf_counter_ns() - start_time)):.3f} "
    f"milliseconds."
)

# --- Plotting ---

# the plot is created
fig, (ax1, ax2) = plt.subplots(
    figsize=(12, 8),
    nrows=2,
    sharex=True,
)

for ax in (ax1, ax2):
    ax.axhline(y=0.0, color="black", linewidth=1.0)

ax1.plot(
    x,
    y,
    label="True signal",
    color="black",
)
ax1.plot(
    x_truncated,
    y_truncated,
    label="Noisy signal to extrapolate",
    color="gray",
)

for iter_i, index_slice in enumerate(
    (
        slice(0, NUM_SAMPLES_EXTRAPOLATED),
        slice(-NUM_SAMPLES_EXTRAPOLATED, None),
    ),
):
    ax1.plot(
        x[index_slice],
        y_extrapolated_burg[index_slice],
        label="Burg extrapolation" if iter_i == 0 else None,
        color="red",
    )
    ax2.plot(
        x[index_slice],
        (y_extrapolated_burg - y)[index_slice],
        color="red",
    )

    ax1.plot(
        x[index_slice],
        y_extrapolated_ols[index_slice],
        label="OLS extrapolation" if iter_i == 0 else None,
        color="blue",
    )
    ax2.plot(
        x[index_slice],
        (y_extrapolated_ols - y)[index_slice],
        color="blue",
    )

ax1.legend(
    ncol=2,
    loc=8,
    bbox_to_anchor=(0.5, 1.015),
)

ax2.set_xlabel("Sample index")
ax1.set_ylabel("Signal value")
ax2.set_ylabel("Extrapolation error")

# the plot is saved ...
if os.getenv("pyscopee_DEVELOPER", "false").lower() == "true":
    plt.savefig(os.path.join(os.path.dirname(__file__), PLOT_FILEPATH))

# ... and shown
plt.show()
