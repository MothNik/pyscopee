"""
Module :mod:`augment.extrapolate._numpy_base`

This module implements NumPy-based basic functions for extrapolating signals beyond
their original range via, e.g.,

- the Burg method for autoregressive model estimation

"""

# === Imports ===

import numpy as np

# === Auxiliary functions ===


def predict_autoregressive_one_side(
    x: np.ndarray,
    ar_coeffs: np.ndarray,
    pad_width: int,
    is_left_side: bool,
) -> np.ndarray:
    """
    Predicts the signal values on one side of the input signal using the coefficients of
    an autoregressive model.

    Parameters
    ----------
    x : :class:`numpy.ndarray` of shape (n,)
        The real input signal for which the extrapolation is to be performed.
    ar_coeffs : :class:`numpy.ndarray` of shape (order + 1,)
        The AR coefficients of the autoregressive model.
        The zero-lag coefficient ``ar_coeffs[0]`` is expected to be present and exactly
        equal to ``1.0``.
    pad_width : :class:`int`
        The size of the extrapolation on the side of the input signal.
        Negative values are silently clipped to ``0``.
    is_left_side : :class:`bool`
        Whether the prediction is for the left side (``True``) or the right side
        of the input signal (``False``) . This distinction is necessary because the
        prediction is performed recursively and the left side requires some additional
        flipping while the right side can be predicted directly.

    Returns
    -------
    x_predicted : :class:`numpy.ndarray` of shape (pad_width,)
        The predicted signal values.

    """

    # if the pad width is <= 0, no prediction is necessary
    if pad_width <= 0:
        return np.array([], dtype=np.float64)

    # the order of the autoregressive model is determined
    order = ar_coeffs.size - 1
    ar_coeffs_internal = np.negative(ar_coeffs[order:0:-1])

    # the output Array is initialised ...
    x_predicted = np.empty(shape=(order + pad_width), dtype=np.float64)
    if is_left_side:
        x_predicted[0:order] = x[order - 1 :: -1]
    else:
        x_predicted[0:order] = x[x.size - order :]

    # ... and the prediction is performed recursively
    for iter_i in range(0, pad_width):
        x_predicted[order + iter_i] = np.sum(
            ar_coeffs_internal * x_predicted[iter_i : order + iter_i]
        )

    # for the left side, the output Array has to be flipped
    if is_left_side:
        return np.flip(x_predicted[order:])
    else:
        return x_predicted[order:]


# === Functions ===


def arburg_slow(
    xs: np.ndarray,
    x_lens: np.ndarray,
    order: int,
) -> np.ndarray:
    def get_x_matrix(x: np.ndarray, order: int) -> np.ndarray:
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
            x_matrix = get_x_matrix(x, iter_ord + 1)
            r_matrix += mat_j @ x_matrix.T @ x_matrix @ mat_j + x_matrix.T @ x_matrix

        k_reflect = -(
            np.append(a, 0.0)
            @ mat_j
            @ r_matrix
            @ np.append(a, 0.0)
            / (np.append(a, 0.0) @ r_matrix @ np.append(a, 0.0))
        )

        a = np.append(a, 0.0) + k_reflect * np.flip(np.append(a, 0.0))
        print(iter_ord, k_reflect, a.tolist())

    return a


def arburg_fast(
    xs: np.ndarray,
    x_lens: np.ndarray,
    order: int,
    tikhonov_lambda: float,
) -> np.ndarray:
    """
    Computes the AR coefficients for an autoregressive model using a fast implementation
    of Burg's method that relies on an implicit matrix formulation that even allows for
    Tikhonov regularisation.

    Parameters
    ----------
    xs : :class:`numpy.ndarray` of shape (m, max(n_i)) of dtype ``numpy.float64``
        The real input signal segments for which the AR coefficients are to be computed.
        Multiple segments are processed by stacking them row-wise in a 2D array whose
        maximum column size is determined by the longest signal. The resulting
        prediction vector will minimise the forward and backward prediction errors
        over all segments combined.
        See ``x_lens`` for the actual Array layout.
    x_lens : :class:`numpy.ndarray` of shape (m,) of dtype ``numpy.int64``
        The lengths of the individual input signal segments.
        ``x_lens[i]`` gives the number of usable elements in ``xs[i, ::]``.
    order : :class:`int`
        The order of the autoregressive model.
    tikhonov_lambda : :class:`float`
        The Tikhonov regularisation parameter lambda. It has to be non-negative
        (``lam >= 0.0``) and if ``> 0.0``, it will result in Tikhonov regularisation.
        Values ``< 0.0`` are silently clipped to ``0.0``.
        Higher values of lambda lead to a more stable solution but may introduce a bias.

    Returns
    -------
    a_prediction : :class:`numpy.ndarray` of shape (order  + 1,)
        The AR coefficients of the autoregressive model.
        To be consistent with Matlab's ``arburg`` function, the zero-lag coefficient is
        included in the output as the first element ``a_prediction[0]`` which is always
        ``1.0``.

    References
    ----------
    The implementation is based on the pseudo-code provided in [1]_ and extended to
    a segmented version using the idea described in [2]_.

    .. [1] Vos K., A Fast Implementation of Burg's Method (2013)
    .. [2] De Waele S., Broersen P.M.T, The Burg Algorithm for Segments, IEEE
       Transactions on Signal Processing (2000), 48(10), pp. 2876-2880,
       DOI: 10.1109/78.869039

    """

    # first, the autocorrelation vectors c would be initialised, but it is more
    # efficient to initialise the auxiliary vectors r with 2 times the autocorrelation
    # values because it has to be updated with a new autocorrelation values in each
    # iteration anyway
    num_segments = x_lens.size
    r_auxiliary = np.zeros(shape=(order + 1, num_segments))
    for iter_i, num_elements in enumerate(x_lens):
        x = xs[iter_i, 0:num_elements]
        for iter_j in range(0, order + 1):
            r_auxiliary[order - iter_j, iter_i] = 2.0 * np.sum(
                x[iter_j:] * x[: num_elements - iter_j]
            )

    r_view = r_auxiliary[order - 1 : order, ::]

    # the penalty is applied if necessary by adding the regularisation parameter to the
    # zero-lag autocorrelation values
    if tikhonov_lambda > 0.0:
        r_auxiliary[order, ::] += tikhonov_lambda

    # then, the reflection and prediction coefficient vectors are initialised ...
    a_prediction = np.zeros(shape=(order + 1))
    a_prediction[0] = 1.0
    a_view = a_prediction[0:1]

    # ... followed by the auxiliary vector g
    g_auxiliary = np.zeros(shape=(order + 1))
    g_view = g_auxiliary[0:2]
    for iter_i, num_elements in enumerate(x_lens):
        g_view[0] += (
            r_auxiliary[order, iter_i]
            - np.square(xs[iter_i, 0])
            - np.square(xs[iter_i, num_elements - 1])
        )
        g_view[1] += r_auxiliary[order - 1, iter_i]

    # the loop for the main recursion is entered
    iter_ord = 0
    for iter_ord in range(0, order - 1):
        # the new reflection coefficient is computed
        k_reflection = -np.sum(a_view * np.flip(g_view)[0 : 1 + iter_ord]) / np.sum(
            a_view * g_view[0 : 1 + iter_ord]
        )

        # then, the Levinson-Durbin recursion is applied to update the prediction
        # coefficients
        a_view = a_prediction[0 : 2 + iter_ord]
        a_view[1 : 1 + iter_ord] += k_reflection * np.flip(a_view[1 : 1 + iter_ord])
        a_view[1 + iter_ord] = k_reflection

        # after that, the auxiliary vectors r and the auxiliary products ΔR @
        # are updated for each segment before they will be summed up in the auxiliary
        # vector g
        g_view += k_reflection * np.flip(g_view)
        r_view_new = r_auxiliary[order - 2 - iter_ord : order, ::]
        for iter_i, num_elements in enumerate(x_lens):
            # the vectors r are updated
            x = xs[iter_i, 0:num_elements]
            r_view[::, iter_i] -= (x[0 : 1 + iter_ord] * x[1 + iter_ord]) + np.flip(
                x[num_elements - 1 - iter_ord : :]
            ) * x[num_elements - 2 - iter_ord]

            # the products ΔR @ are computed
            # ΔR is a rank-1 matrix, but it is more efficient to compute the individual
            # vector-vector products
            x_view = np.flip(x[0 : 2 + iter_ord])
            delta_r_dot_a = -x_view * np.sum(x_view * a_view)
            x_view = x[num_elements - 2 - iter_ord : :]
            delta_r_dot_a -= x_view * np.sum(x_view * a_view)

            # the auxiliary vector g is updated
            g_view += delta_r_dot_a
            g_auxiliary[2 + iter_ord] += np.sum(r_view_new[::, iter_i] * a_view)

        # the views of the auxiliary vectors are updated
        r_view = r_view_new
        g_view = g_auxiliary[0 : 3 + iter_ord]

    # the last update of the reflection and prediction coefficients is performed
    iter_ord += 1
    k_reflection = -np.sum(a_view * np.flip(g_view)[0 : 1 + iter_ord]) / np.sum(
        a_view * g_view[0 : 1 + iter_ord]
    )
    a_view = a_prediction[0 : 2 + iter_ord]
    a_view[1 : 1 + iter_ord] += k_reflection * np.flip(a_view[1 : 1 + iter_ord])
    a_view[1 + iter_ord] = k_reflection

    return a_prediction


def extrapolate_autoregressive(
    x: np.ndarray,
    ar_coeffs: np.ndarray,
    pad_width_left: int,
    pad_width_right: int,
) -> np.ndarray:
    """
    Extrapolates a signal beyond its original range using the coefficients of an
    autoregressive model.

    Parameters
    ----------
    x : :class:`numpy.ndarray` of shape (n,)
        The real input signal to be extrapolated.
    ar_coeffs : :class:`numpy.ndarray` of shape (order + 1,)
        The AR coefficients of the autoregressive model.
        The zero-lag coefficient ``ar_coeffs[0]`` is expected to be present and exactly
        equal to ``1.0``.
    pad_width_left, pad_width_right : :class:`int`
        The size of the extrapolation on the left and right side of the input signal,
        respectively. Negative values are silently clipped to ``0``, which means that no
        extrapolation is performed on the respective side.

    Returns
    -------
    x_extrapolated : :class:`numpy.ndarray` of shape (n + pad_left + pad_right,)
        The extrapolated signal.

    """

    return np.concatenate(
        (
            predict_autoregressive_one_side(
                x=x,
                ar_coeffs=ar_coeffs,
                pad_width=pad_width_left,
                is_left_side=True,
            ),
            x,
            predict_autoregressive_one_side(
                x=x,
                ar_coeffs=ar_coeffs,
                pad_width=pad_width_right,
                is_left_side=False,
            ),
        )
    )


if __name__ == "__main__":
    import numpy as np
    from scipy.signal import lfilter

    # Set the random seed for reproducibility
    np.random.seed(1)

    # Define the filter coefficients
    A = [1, -2.7607, 3.8106, -2.6535, 0.9238, 1.11111]

    # Generate random noise
    noise = 0.2 * np.random.rand(1024)
    print(noise)

    # Apply the filter
    y = lfilter(1, A, noise)

    # Now, `y` contains the filtered output
    print(arburg_slow(y[np.newaxis, ::], np.array([y.size]), 5))  # type: ignore
    print(arburg_fast(y[np.newaxis, ::], np.array([y.size]), 5, 0.0))  # type: ignore