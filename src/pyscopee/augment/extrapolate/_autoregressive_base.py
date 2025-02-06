"""
Module :mod:`augment.extrapolate._autoregressive_base`

This module implements functions for extrapolating signals beyond their original range
using autoregressive models, such as

- the (segmented, Tikhonov-regularised) Burg method for AR coefficient estimation
- the (segmented, Tikhonov-regularised) Ordinary Least Squares (OLS
    coefficient estimation

"""

# === Imports ===

from typing import Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve as scipy_solve

from ..._utils import jit

# === Auxiliary functions ===


@jit(
    "float64[:](float64[:], float64[:], int64, boolean)",
    nopython=True,
    cache=True,
)
def predict_autoregressive_one_side(
    x: NDArray[np.float64],
    ar_coeffs: NDArray[np.float64],
    pad_width: int,
    is_left_side: bool,
) -> NDArray[np.float64]:
    """
    Predicts the signal values on one side of the input signal using the coefficients of
    an autoregressive model.

    Parameters
    ----------
    x : :class:`numpy.ndarray` of shape (n,)  of dtype ``numpy.float64``
        The real input signal for which the extrapolation is to be performed.
    ar_coeffs : :class:`numpy.ndarray` of shape (order + 1,)  of dtype ``numpy.float64``
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
    x_predicted : :class:`numpy.ndarray` of shape (pad_width,)  of dtype ``numpy.float64``
        The predicted signal values.

    """  # noqa: E501

    # if the pad width is <= 0, no prediction is necessary
    if pad_width <= 0:
        return np.empty(shape=(0,), dtype=np.float64)

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
        x_predicted[order + iter_i] = np.dot(
            ar_coeffs_internal,
            x_predicted[iter_i : order + iter_i],
        )

    # for the left side, the output Array has to be flipped
    if is_left_side:
        return np.flip(x_predicted[order:])
    else:
        return x_predicted[order:]


# === Functions ===


@jit(
    "float64[:](float64[:,:], int64[:], int64, float64)",
    nopython=True,
    cache=True,
)
def arburg_fast(
    xs: NDArray[np.float64],
    x_lens: NDArray[np.int64],
    order: int,
    tikhonov_lambda: float,
) -> NDArray[np.float64]:
    """
    Computes the AR coefficients for an autoregressive model using a fast implementation
    of Burg's method that relies on an implicit matrix formulation that even allows for
    Tikhonov regularisation.

    Parameters
    ----------
    xs : :class:`numpy.ndarray` of shape (m, max(n_i)) of dtype ``numpy.float64``
        The real input signal segments for which the AR coefficients are to be computed.
        Multiple segments are processed by stacking them row-wise in a 2D array whose
        maximum column size is determined by the longest segment. The resulting
        prediction vector will minimise the forward and backward prediction errors
        over all segments combined (but not across segments).
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
    a_prediction : :class:`numpy.ndarray` of shape (order  + 1,) of dtype ``numpy.float64``
        The AR coefficients of the autoregressive model.
        To be consistent with Matlab's ``arburg`` function, the zero-lag coefficient is
        included in the output as the first element ``a_prediction[0]`` which is always
        ``1.0``.
        Its ``i``-th element corresponds to the coefficient of the ``i``-th lag
        starting from ``0`` for the zero-lag coefficient.

    References
    ----------
    The implementation is based on the pseudo-code provided in [1]_ and extended to
    a segmented version using the idea described in [2]_.

    .. [1] Vos K., A Fast Implementation of Burg's Method (2013)
    .. [2] De Waele S., Broersen P.M.T, The Burg Algorithm for Segments, IEEE
       Transactions on Signal Processing (2000), 48(10), pp. 2876-2880,
       DOI: 10.1109/78.869039

    """  # noqa: E501

    # first, the autocorrelation vectors c would be initialised, but it is more
    # efficient to initialise the auxiliary vectors r with 2 times the autocorrelation
    # values because it has to be updated with a new autocorrelation values in each
    # iteration anyway
    num_segments = x_lens.size
    r_auxiliary = np.zeros(shape=(order + 1, num_segments))
    for iter_i, num_elements in enumerate(x_lens):
        x = xs[iter_i, 0:num_elements]
        for iter_j in range(0, order + 1):
            r_auxiliary[order - iter_j, iter_i] = (
                2.0
                * np.correlate(
                    x[iter_j:],
                    x[: num_elements - iter_j],
                    mode="valid",
                )[0]
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

    # ... followed by the auxiliary vector g which resembles the product of the
    # correlation matrix R and the prediction coefficients a
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

        # after that, the auxiliary vectors r and the auxiliary products ΔR @ a
        # are updated for each segment before they will be summed up in the auxiliary
        # vector g
        # NOTE: ΔR is a rank-1 update matrix
        g_view += k_reflection * np.flip(g_view)
        r_view_new = r_auxiliary[order - 2 - iter_ord : order, ::]
        for iter_i, num_elements in enumerate(x_lens):
            # the vectors r are updated
            x = xs[iter_i, 0:num_elements]
            r_view[::, iter_i] -= (x[0 : 1 + iter_ord] * x[1 + iter_ord]) + np.flip(
                x[num_elements - 1 - iter_ord : :]
            ) * x[num_elements - 2 - iter_ord]

            # the products ΔR @ a are computed
            # ΔR is a rank-1 matrix, but it is more efficient to compute the individual
            # vector-vector products with the vector a directly
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


# TODO: make this more efficient by adding the partial X.T @ X directly to the left and
#       right hand side of the least-squares system rather than computing the full
#       X.T matrix first before the matrix multiplication
@jit(
    "Tuple((float64[:,:], float64[:]))(float64[:,:], int64[:], int64, int64, float64)",
    nopython=True,
    cache=True,
)
def _make_ar_one_step_least_squares_system(
    xs: NDArray[np.float64],
    x_lens: NDArray[np.int64],
    order: int,
    num_equations: int,
    tikhonov_lambda: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Constructs the left-hand side ``A`` and right-hand side vector ``b`` of the
    least-squares ``A @ x ~= b`` system for the one-step-ahead autoregressive model with
    the coefficients ``x``.

    Here,

    - ``A`` is given by ``X.T @ X + lambda * I`` where ``X`` is the design matrix of the
        autoregressive model, ``lambda`` is the Tikhonov regularisation parameter, and
        ``I`` is the identity matrix
    - ``b`` is given by ``X.T @ y`` where ``y`` is the target vector of the
        autoregressive model

    Here, the right hand side ``b`` is just a column vector because only the next
    prediction is computed (i.e., the one-step-ahead model).

    Parameters
    ----------
    xs : :class:`numpy.ndarray` of shape (m, max(n_i)) of dtype ``numpy.float64``
        The real input signal segments for which the AR coefficients are to be computed.
        Multiple segments are processed by stacking them row-wise in a 2D array whose
        maximum column size is determined by the longest segment. The resulting
        prediction vector will minimise the forward and backward prediction errors
        over all segments combined (but not across segments).
        See ``x_lens`` for the actual Array layout.
    x_lens : :class:`numpy.ndarray` of shape (m,) of dtype ``numpy.int64``
        The lengths of the individual input signal segments.
        ``x_lens[i]`` gives the number of usable elements in ``xs[i, ::]``.
    order : :class:`int`
        The order of the autoregressive model.
    num_equations : :class:`int`
        The number of equations in the least-squares system.
    tikhonov_lambda : :class:`float`
        The Tikhonov regularisation parameter lambda. It has to be non-negative
        (``lam >= 0.0``) and if ``> 0.0``, it will result in Tikhonov regularisation.
        Values ``< 0.0`` are silently clipped to ``0.0``.
        Higher values of lambda lead to a more stable solution but may introduce a bias.

    Returns
    -------
    lhs_matrix : :class:`numpy.ndarray` of shape (order, order) of dtype ``numpy.float64``
        The (regularized) left-hand side matrix of the least-squares system.
    rhs_vector : :class:`numpy.ndarray` of shape (order,) of dtype ``numpy.float64``
        The right-hand side vector of the least-squares system.

    """  # noqa: E501

    # the left-hand side matrices need to be concatenated by using sliding window views
    # of the input signal segments, each window of size ``order``
    lhs_matrix = np.empty(shape=(num_equations, order), dtype=np.float64)
    # the right-hand side matrix is simply the input signal segments with the first
    # ``order`` elements removed
    rhs_vector = np.empty(shape=(num_equations,), dtype=np.float64)

    # the left and right hand side are filled by means of a simple loop for the forward
    # predictions
    row_index_from = 0
    for iter_i, num_elements in enumerate(x_lens):
        row_index_to = row_index_from + num_elements - order
        lhs_matrix[row_index_from:row_index_to, ::] = (
            np.lib.stride_tricks.sliding_window_view(
                xs[iter_i, 0 : num_elements - 1],
                window_shape=(order,),
            )
        )

        rhs_vector[row_index_from:row_index_to] = xs[iter_i, order:num_elements]

        row_index_from = row_index_to

    # now, the process is repeated for the backward predictions
    for iter_i, num_elements in enumerate(x_lens):
        row_index_to = row_index_from + num_elements - order
        lhs_matrix[row_index_from:row_index_to, ::] = (
            np.lib.stride_tricks.sliding_window_view(
                np.flip(xs[iter_i, 1:num_elements]),
                window_shape=(order,),
            )
        )

        rhs_vector[row_index_from:row_index_to] = np.flip(
            xs[iter_i, 0 : num_elements - order]
        )

        row_index_from = row_index_to

    # finally, the normal equations are formed by first computing the right-hand side
    # ``b = X.T @ y``
    rhs_vector = lhs_matrix.T @ rhs_vector
    # then, the left-hand side is updated for the normal equations to
    # ``A.T @ A + lambda * I``
    lhs_matrix = lhs_matrix.T @ lhs_matrix
    if tikhonov_lambda > 0.0:
        np.fill_diagonal(lhs_matrix, np.diag(lhs_matrix) + tikhonov_lambda)

    return lhs_matrix, rhs_vector


def ar_one_step_least_squares(
    xs: NDArray[np.float64],
    x_lens: NDArray[np.int64],
    order: int,
    num_equations: int,
    tikhonov_lambda: float,
    lstsq_solver: Literal["sym", "pos"],
) -> NDArray[np.float64]:
    """
    Computes the AR coefficients for a one-step-ahead autoregressive model using a
    an Ordinary Least Squares (OLS) approach with optional Tikhonov regularisation.

    Parameters
    ----------
    xs : :class:`numpy.ndarray` of shape (m, max(n_i)) of dtype ``numpy.float64``
        The real input signal segments for which the AR coefficients are to be computed.
        Multiple segments are processed by stacking them row-wise in a 2D array whose
        maximum column size is determined by the longest segment. The resulting
        prediction vector will minimise the forward and backward prediction errors
        over all segments combined (but not across segments).
        See ``x_lens`` for the actual Array layout.
    x_lens : :class:`numpy.ndarray` of shape (m,) of dtype ``numpy.int64``
        The lengths of the individual input signal segments.
        ``x_lens[i]`` gives the number of usable elements in ``xs[i, ::]``.
    order : :class:`int`
        The order of the autoregressive model.
    num_equations : :class:`int`
        The number of equations in the least-squares system.
    tikhonov_lambda : :class:`float`
        The Tikhonov regularisation parameter lambda. It has to be non-negative
        (``lam >= 0.0``) and if ``> 0.0``, it will result in Tikhonov regularisation.
        Values ``< 0.0`` are silently clipped to ``0.0``.
        Higher values of lambda lead to a more stable solution but may introduce a bias.
        A value of ``0.0`` corresponds to the standard OLS approach, but this may lead
        to numerical instability.
    lstsq_solver : {``"sym"``, ``"pos"``}
        The solver to use for the least squares problem, which can be

        - ``"sym"``: Symmetric indefinite factorisation which is a slower but more
            stable solver.
        - ``"pos"``: Cholesky factorisation which is the a very fast but less stable
            solver.

    Returns
    -------
    a_prediction : :class:`numpy.ndarray` of shape (order  + 1,) of dtype ``numpy.float64``
        The AR coefficients of the autoregressive model.
        To be consistent with Matlab's ``arburg`` function, the zero-lag coefficient is
        included in the output as the first element ``a_prediction[0]`` which is always
        ``1.0``.
        Its ``i``-th element corresponds to the coefficient of the ``i``-th lag
        starting from ``0`` for the zero-lag coefficient.

    Raises
    -------
    numpy.linalg.LinAlgError
        If the least-squares system is singular and cannot be solved with the given
        ``tikhonov_lambda``.

    """  # noqa: E501

    # the left-hand side matrix ``A`` and right-hand side vector ``b`` of the
    # least-squares system are constructed
    lhs_matrix, rhs_vector = _make_ar_one_step_least_squares_system(
        xs=xs,
        x_lens=x_lens,
        order=order,
        num_equations=num_equations,
        tikhonov_lambda=tikhonov_lambda,
    )

    # the AR coefficients are computed by solving the (regularized) least-squares system
    # NOTE: the addition of the zero-lag coefficient, the flip, and the sign flipping is
    #       required due to the conventions for the autoregressive coefficients used by
    #       Matlab's ``arburg`` function
    a_prediction = np.empty(shape=(order + 1), dtype=np.float64)
    a_prediction[0] = 1.0

    # an attempt is made to solve the system directly with the given regularisation
    # parameter, but this may fail if the system is singular since a dedicated solver
    a_prediction[1:] = np.negative(
        np.flip(
            scipy_solve(
                a=lhs_matrix,
                b=rhs_vector,
                assume_a=lstsq_solver,
            )
        )
    )

    return a_prediction


@jit(
    "float64[:](float64[:], float64[:, :], int64, int64, int64, int64)",
    nopython=True,
    cache=True,
)
def extrapolate_autoregressive(
    x: NDArray[np.float64],
    ar_coeffs: NDArray[np.float64],
    ar_order_left: int,
    ar_order_right: int,
    pad_width_left: int,
    pad_width_right: int,
) -> NDArray[np.float64]:
    """
    Extrapolates a signal beyond its original range using the coefficients of an
    autoregressive model.

    Parameters
    ----------
    x : :class:`numpy.ndarray` of shape (n,)  of dtype ``numpy.float64``
        The real input signal to be extrapolated.
    ar_coeffs : :class:`numpy.ndarray` of shape (2, max(ar_order_left, ar_order_right) + 1)  of dtype ``numpy.float64``
        The AR coefficients of the autoregressive model.
        Its first row and second row correspond to the AR coefficients for the left and
        right side, respectively. Please refer to the Notes section for more details.
        The zero-lag coefficients ``ar_coeffs[::, 0]`` is expected to be present and
        exactly equal to ``1.0``.
        Its ``i``-th column has to correspond to the coefficients of the ``i``-th lag.
    ar_order_left, ar_order_right : :class:`int`
        The order of the autoregressive model for the left and right side of the input
        signal, respectively. Please refer to the Notes section for more details.
    pad_width_left, pad_width_right : :class:`int`
        The size of the extrapolation on the left and right side of the input signal,
        respectively. Negative values are silently clipped to ``0``, which means that no
        extrapolation is performed on the respective side.

    Returns
    -------
    x_extrapolated : :class:`numpy.ndarray` of shape (n + pad_left + pad_right,)  of dtype ``numpy.float64``
        The extrapolated signal.

    Notes
    -----
    The AR coefficients - including the zero-lag coefficient - for the left hand side
    and the right hand side can be accessed as follows:


    ```python
    ar_coeffs_left = ar_coeffs[0, 0:ar_order_left + 1]
    ar_coeffs_right = ar_coeffs[1, 0:ar_order_right + 1]
    ```

    """  # noqa: E501

    return np.concatenate(
        (
            predict_autoregressive_one_side(
                x=x,
                ar_coeffs=ar_coeffs[0, 0 : ar_order_left + 1],
                pad_width=pad_width_left,
                is_left_side=True,
            ),
            x,
            predict_autoregressive_one_side(
                x=x,
                ar_coeffs=ar_coeffs[1, 0 : ar_order_right + 1],
                pad_width=pad_width_right,
                is_left_side=False,
            ),
        )
    )
