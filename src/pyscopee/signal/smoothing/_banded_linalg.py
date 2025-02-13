"""
Module :mod:`signal.smoothing._banded_linalg`

This module provides interfaces to LAPACK-routines for banded matrices such as

- LU decomposition of banded matrices

Besides, is also augments the LAPACK-routines with additional functionality such as

- computing the determinant of a banded matrix using its LU decomposition

"""

# === Setup ===

__all__ = [
    "BandedLUFactorization",
    "lu_banded",
    "lu_solve_banded",
    "slogdet_lu_banded",
]

# === Imports ===

from collections import namedtuple
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import lapack

# === Typing ===

LAndUBandCounts = Tuple[int, int]
BandedLUFactorization = namedtuple(
    "BandedLUFactorization",
    [
        "lub",
        "ipiv",
        "l_and_u",
        "singular",
    ],
)


# === Banded LU decomposition ===


def lu_banded(
    l_and_u: LAndUBandCounts,
    ab: ArrayLike,
    *,
    check_finite: bool = True,
    ab_has_added_workspace: bool = False,
) -> BandedLUFactorization:
    """
    Computes the LU decomposition of a banded matrix ``A`` using LAPACK-routines.
    This function is a wrapper of the LAPACK-routine ``?gbtrf`` which computes the LU
    decomposition of a banded matrix ``A`` in-place. It wraps the routine in an
    analogous way to SciPy's :func:`scipy.linalg.cholesky_banded`.

    Parameters
    ----------
    l_and_u : (:obj:`int`, :obj:`int`)
        The number of non-zero sub- and superdiagonals of the matrix ``A``,
        respectively.
    ab : Array_like of shape (l_and_u[0] + 1 + l_and_u[1], n) or (2 * l_and_u[0] + 1 + l_and_u[1], n)
        A 2D-Array resembling the matrix ``A`` in banded storage format (see Notes).
        Its expected shape depends on ``ab_has_added_workspace``.
    check_finite : bool, default=True
        Whether to check that the input matrix contains only finite numbers. Disabling
        may give a performance gain, but may result in problems (crashes,
        non-termination) if the inputs do contain infinities or NaNs.
    ab_has_added_workspace : :obj:`bool`, default=``False``
        Whether ``ab`` already has the additional workspace for pivoting added to it
        (``True``) or not (``False``). If ``False``, the workspace will be added
        internally.
        If ``False``, the number of shape of ``ab`` is expected to be
        ``(l_and_u[0] + 1 + l_and_u[1], n)``.
        If ``True``, it is expected that ``l_and_u[0]`` leading rows are added to
        ``ab``, so its shape must be ``(2 * l_and_u[0] + 1 + l_and_u[1], n)``.

    Returns
    -------
    lub : :obj:`numpy.ndarray` of shape (2 * l_and_u[0] + 1 + l_and_u[1], n)
        The LU decomposition of the matrix ``A`` in banded storage format (see Notes).
    ipiv : :obj:`numpy.ndarray` of shape (n,)
        The pivoting indices that define the permutation matrix ``P``.
    l_and_u : (:obj:`int`, :obj:`int`)
        The number of sub- and superdiagonals of the matrix ``A`` that are non-zero.
    singular : :obj:`bool`
        A boolean indicating whether the matrix is singular.

    Notes
    -----
    For LAPACK's banded LU decomposition, the matrix ``a`` is stored in ``ab`` using the
    matrix diagonal ordered form:

        ```python
        ab[u + i - j, j] == a[i,j] # see below for u
        ```

    An example of ``ab`` (shape of a is ``(7,7)``, ``u``=3 superdiagonals, ``l``=2
    subdiagonals) looks like:

        ```python
             *    *    *   a03  a14  a25  a36
             *    *   a02  a13  a24  a35  a46
             *   a01  a12  a23  a34  a45  a56   # ^ superdiagonals
            a00  a11  a22  a33  a44  a55  a66   # main diagonal
            a10  a21  a32  a43  a54  a65   *    # v subdiagonals
            a20  a31  a42  a53  a64   *    *
        ```

    where all entries marked with `*` are zero elements although they will be set to
    arbitrary values by this function.

    Internally LAPACK relies on an expanded version of this format to perform inplace
    operations that adds another ``l`` superdiagonals to the matrix in order to
    overwrite them for the purpose of pivoting. The output is thus an expanded version
    of the LU decomposition of ``A`` in the same format where the main diagonal of
    ``L`` is implicitly taken to be a vector of ones. The output can directly be used
    for the LAPACK-routine ``gbtrs`` to solve linear systems of equations based on this
    decomposition.

    """

    # the (optional) finite check and Array-conversion are performed
    if check_finite:
        ab = np.asarray_chkfinite(ab)
    else:
        ab = np.asarray(ab)

    # then, the number of lower and upper subdiagonals needs to be checked for being
    # consistent with the shape of ``ab``
    num_subdiagonals, num_superdiagonals = l_and_u
    required_num_rows = (
        (1 + int(ab_has_added_workspace)) * num_subdiagonals + 1 + num_superdiagonals
    )

    if ab.shape[0] != required_num_rows:  # pragma: no cover
        raise ValueError(
            f"\nInvalid values for the number of sub- and super "
            f"diagonals: l+u+1 ({num_subdiagonals + num_superdiagonals + 1}) does not "
            f"equal ab.shape[0] ({ab.shape[0]})."
        )

    # now, the LAPACK-routines can be called
    # to make ``ab`` compatible with the shape the LAPACK expects in this case, it
    # needs to be re-written into a larger Array that has zeros elsewhere
    # FIXME: for tridiagonal matrices, the SciPy wrapper for ``gttrf`` should be used
    lapack_routine = "gbtrf"
    (gbtrf,) = lapack.get_lapack_funcs((lapack_routine,), (ab,))
    if ab_has_added_workspace:
        lpkc_ab = ab

    else:
        lpkc_ab = np.row_stack(
            (
                np.zeros((num_subdiagonals, ab.shape[1]), dtype=ab.dtype),
                ab,
            )
        )

    lub, ipiv, info = gbtrf(
        ab=lpkc_ab,
        kl=num_subdiagonals,
        ku=num_superdiagonals,
    )

    # then, the results needs to be validated and returned
    # Case 1: the factorisation could be completed, which does not imply that the
    # solution can be used for solving a linear system
    if info >= 0:
        return BandedLUFactorization(
            lub=lub,
            ipiv=ipiv,
            l_and_u=l_and_u,
            singular=info > 0,
        )

    # Case 2: the factorisation was not completed due to invalid input
    raise ValueError(  # pragma: no cover # noqa: E501
        f"\nIllegal value in {-info}-th argument of internal gbtrf."
    )


def lu_solve_banded(
    lub_factorization: BandedLUFactorization,
    b: np.ndarray,
) -> np.ndarray:
    """
    Solves a linear system of equations ``Ax=b`` with a banded matrix ``A`` using its
    precomputed LU decomposition.
    This function wraps the LAPACK-routine ``gbtrs`` in an analogous way to SciPy's
    ``scipy.linalg.cho_solve_banded``.

    Parameters
    ----------
    lub_factorization : BandedLUFactorization
        The LU decomposition of the matrix ``A`` in banded storage format as returned by
        the function :func:`lu_banded`.
    b : ndarray of shape (n,)
        A 1D-Array containing the right-hand side of the linear system of equations.

    Returns
    -------
    x : ndarray of shape (n,)
        The solution to the system ``A x = b``.

    Raises
    ------
    LinAlgError
        If the system to solve is singular.

    """

    # if the matrix is singular, the solution cannot be computed
    if lub_factorization.singular:
        raise np.linalg.LinAlgError("\nSystem is singular.")

    # then, the shapes of the LU decomposition and ``b`` need to be validated against
    # each other
    if lub_factorization.lub.shape[1] != b.shape[0]:  # pragma: no cover
        raise ValueError(
            f"\nShapes of lub ({lub_factorization.lub.shape[1]}) and b "
            f"({b.shape[0]}) are not compatible."
        )

    # now, the LAPACK-routine is called
    (gbtrs,) = lapack.get_lapack_funcs(("gbtrs",), (lub_factorization.lub, b))
    x, info = gbtrs(
        ab=lub_factorization.lub,
        kl=lub_factorization.l_and_u[0],
        ku=lub_factorization.l_and_u[1],
        b=b,
        ipiv=lub_factorization.ipiv,
    )

    # then, the results needs to be validated and returned
    # Case 1: the solution could be computed truly successfully, i.e., without any
    # NaN-values
    if info == 0 and not np.isnan(x).any():
        return x

    # Case 2: the solution was computed, but there were NaN-values in it
    elif info == 0:
        raise np.linalg.LinAlgError("\nMatrix is singular.")

    # Case 3: the solution could not be computed due to invalid input
    elif info < 0:  # pragma: no cover
        raise ValueError(f"\nIllegal value in {-info}-th argument of internal gbtrs.")

    # Case 4: unexpected error
    raise AssertionError(  # pragma: no cover
        f"\nThe internal gbtrs returned info > 0 ({info}) which should not happen."
    )


def slogdet_lu_banded(
    lub_factorization: BandedLUFactorization,
) -> tuple[float, float]:
    """
    Computes the logarithm of the absolute value and the sign of the determinant of a
    banded matrix A using its LU decomposition. This is way more efficient than
    computing the determinant directly because the LU decompositions main diagonals
    already encode the determinant as the product of the diagonal entries of the
    factors.

    Parameters
    ----------
    lub_factorization : BandedLUFactorization
        The LU decomposition of the matrix ``A`` in banded storage format as returned by
        the function :func:`lu_banded`.

    Returns
    -------
    sign : float
        A number representing the sign of the determinant.
    logabsdet : float
        The natural log of the absolute value of the determinant.
        If the determinant is zero, then `sign` will be 0 and `logabsdet` will be
        -Inf. In all cases, the determinant is equal to ``sign * np.exp(logabsdet)``.

    Raises
    ------
    OverflowError
        If any of the diagonal entries of the LU decomposition leads to an overflow in
        the natural logarithm.

    """

    # first, the number of actual row exchanges needs to be counted
    unchanged_row_indices = np.arange(
        start=0,
        stop=lub_factorization.lub.shape[1],
        step=1,
        dtype=lub_factorization.ipiv.dtype,
    )
    num_row_exchanges = np.count_nonzero(
        lub_factorization.ipiv != unchanged_row_indices
    )

    # the sign-prefactor of the determinant is either +1 or -1 depending on whether the
    # number of row exchanges is even or odd
    sign = -1.0 if num_row_exchanges % 2 == 1 else 1.0

    # since the determinant (without sign prefactor) is just the product of the diagonal
    # product of L and the diagonal product of U, the calculation simplifies. As the
    # main diagonal of L is a vector of ones, only the diagonal product of U is required
    main_diagonal_row_index = (
        lub_factorization.lub.shape[0] - 1 - lub_factorization.l_and_u[0]
    )
    main_diagonal = lub_factorization.lub[main_diagonal_row_index, ::]
    u_diagonal_sign_is_positive = np.count_nonzero(main_diagonal < 0.0) % 2 == 0
    with np.errstate(divide="ignore", over="ignore"):
        logabsdet = np.log(np.abs(main_diagonal)).sum()

    # logarithms of zero are already properly handled, so there is not reason to worry
    # about, since they are -inf which will result in a zero determinant in exp();
    # overflow however needs to lead to a raise and in this case the log(det) is either
    # +inf in case of overflow only or NaN in case of the simultaneous occurrence of
    # zero and overflow
    if np.isnan(logabsdet) or np.isposinf(logabsdet):  # pragma: no cover
        raise OverflowError(
            "\nFloating point overflow in natural logarithm. At least 1 main diagonal "
            "entry results in overflow, thereby corrupting the determinant."
        )

    # finally, the absolute value of the natural logarithm of the determinant is
    # returned together with its sign
    if np.isneginf(logabsdet):  # pragma: no cover
        return 0.0, logabsdet

    if u_diagonal_sign_is_positive:
        return sign, logabsdet

    return -sign, logabsdet
