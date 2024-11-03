"""
This example script shows the :func:`arburg` function implemented in ``pyscopee``
when applied to a single signal.

"""

# === Imports ===

import os

import numpy as np
from matplotlib import pyplot as plt

from pyscopee.augment import arburg, extrapolate_autoregressive

plt.style.use(os.path.join(os.path.dirname(__file__), "./pyscopee.mplstyle"))

# === Constants ===

# the size of the signal
SIGNAL_SIZE = 1000
# the order of the autoregressive model
AR_ORDER = 800
# the number of samples to extrapolate
NUM_SAMPLES_EXTRAPOLATED = 2000

# the path to save the plot
PLOT_FILEPATH = "./example_plots/02_burg_autoregression_single_segment.png"


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

# the AR coefficients are computed
y_truncated = y[NUM_SAMPLES_EXTRAPOLATED:-NUM_SAMPLES_EXTRAPOLATED]
ar_coeffs = arburg(
    xs=y_truncated,
    order=AR_ORDER,
    tikhonov_lambda=1e0,
    jit=True,
)


# the signal is extrapolated
y_extrapolated = extrapolate_autoregressive(
    x=y_truncated,
    ar_coeffs=ar_coeffs,
    pad_width=(NUM_SAMPLES_EXTRAPOLATED, NUM_SAMPLES_EXTRAPOLATED),
)

# the plot is created
fig, ax = plt.subplots(
    figsize=(12, 8),
)

ax.plot(
    x,
    y,
    label="Original signal",
    color="black",
)
ax.plot(
    x,
    y_extrapolated,
    label="Extrapolated signal",
    color="red",
)

ax.legend()

ax.set_xlabel("Sample index")
ax.set_ylabel("Signal value")

plt.show()
