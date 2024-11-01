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


def arburg_slow(
    xs: np.ndarray,
    x_lens: np.ndarray,
    order: int,
) -> np.ndarray:
    """
    This function implements the (segmented) Burg method for autoregressive model
    estimation using a slow but very literal implementation.

    For the input arguments, please refer to the documentation of the function
    :func:`pyscopee.augment.extrapolate._numpy_base.arburg_fast`.

    """

    # a nested function to shape the x-data into a matrix with lagged columns
    def shape_to_lagged_x_matrix(
        x: np.ndarray,
        order: int,
    ) -> np.ndarray:
        x_matrix = np.empty(shape=(x.size - order, order + 1), dtype=np.float64)
        for iter_i in range(0, order + 1):
            x_matrix[::, order - iter_i] = x[iter_i : x.size - order + iter_i]

        return x_matrix

    a = np.array([1.0])
    for iter_ord in range(0, order):
        mat_j = np.flip(np.eye(iter_ord + 2), axis=1)
        r_matrix = np.zeros(shape=(iter_ord + 2, iter_ord + 2))
        for iter_i, num_elements in enumerate(x_lens):
            x = xs[iter_i, 0:num_elements]
            x_matrix = shape_to_lagged_x_matrix(x, iter_ord + 1)
            r_matrix += mat_j @ x_matrix.T @ x_matrix @ mat_j + x_matrix.T @ x_matrix

        k_reflect = -(
            np.append(a, 0.0)
            @ mat_j
            @ r_matrix
            @ np.append(a, 0.0)
            / (np.append(a, 0.0) @ r_matrix @ np.append(a, 0.0))
        )

        a = np.append(a, 0.0) + k_reflect * np.flip(np.append(a, 0.0))

    return a


# === Main ===

# the signal is set up
"""
vector<double> original( 128, 0.0 );
for ( size_t i = 0; i < original.size(); i++ )
 {
 original[ i ] = cos( i * 0.01 ) + 0.75 *cos( i * 0.03 )
 + 0.5 *cos( i * 0.05 ) + 0.25 *cos( i * 0.11 );
 }
 // GET LINEAR PREDICTION COEFFICIENTS
 vector<double> coeffs( 4, 0.0 )
"""

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
    tikhonov_lambda=1e-5,
    jit=True,
)
print(ar_coeffs)
"""
ar_coeffs = arburg_slow(
    xs=np.array([y_truncated]),
    x_lens=np.array([y_truncated.size]),
    order=AR_ORDER,
)
"""

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
