"""
Module :mod:`linalg._banded_linalg`

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
from typing import List, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import lapack

from .._utils import (
    Integer,
    NumPyDTypeKinds,
    get_validated_integer,
    get_validated_numeric_nd_array_like,
)

# === Typing ===

_Scalar = Union[int, float, complex, np.integer, np.floating, np.complexfloating]
LAndUBandCounts = Union[Tuple[Integer, Integer], List[Integer]]
BandedLUFactorization = namedtuple(
    "BandedLUFactorization",
    [
        "lub",
        "ipiv",
        "l_and_u",
        "singular",
    ],
)

# === Auxiliary Functions ===


def _is_data_linked(
    arr: np.ndarray,
    original: Union[_Scalar, ArrayLike],
) -> bool:
    """
    Strictly checks for ``arr`` not sharing any data with ``original``, under the
    assumption that ``arr = atleast_1d(original)`` followed by a potential type
    conversion.
    If ``arr`` is a view of ``original``, this function returns ``False``.

    Was copied from the SciPy utility function ``scipy.linalg._misc._datacopied``, but
    the name and the docstring were adapted to make them clearer. Besides, the check for
    scalar ``original``s was added.

    """

    if np.isscalar(original):
        return False
    if arr is original:
        return True
    if not isinstance(original, np.ndarray) and hasattr(original, "__array__"):
        return original.__array__().dtype is arr.dtype  # type: ignore

    return arr.base is not None


# === Banded LU decomposition ===


def lu_banded(
    l_and_u: LAndUBandCounts,
    ab: ArrayLike,
    *,
    overwrite_ab: bool = False,
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
        Both have to non-negative ``>= 0``.
    ab : Array_like of shape (l_and_u[0] + 1 + l_and_u[1], n) or (2 * l_and_u[0] + 1 + l_and_u[1], n)
        A 2D-Array resembling the matrix ``A`` in banded storage format (see Notes).
        Its expected number of rows depends on ``ab_has_added_workspace`` while its
        number of columns has to be ``>= max(l_and_u) + 1``.
    overwrite_ab : bool, default=``False``
        Whether the input matrix ``ab`` is allowed to be overwritten by the routine
        (``True``) or not (``False``).
        This only has an effect if ``ab_has_added_workspace`` is ``True``.
    check_finite : bool, default=``True``
        Whether to check that the input matrix contains only finite numbers. Disabling
        may give a performance gain, but may result in problems (crashes,
        non-termination) if the inputs do contain infinities or NaNs.
    ab_has_added_workspace : :obj:`bool`, default=``False``
        Whether ``ab`` already has the additional workspace for pivoting added to it
        (``True``) or not (``False``). If ``False``, the workspace will be added
        internally.
        An additional workspace of ``l_and_u[0]`` superdiagonals is required for
        pivoting. Therefore, the expected shape of ``ab`` is

        - ``False``: ``(l_and_u[0] + 1 + l_and_u[1], n)``
        - ``True``: ``(2 * l_and_u[0] + 1 + l_and_u[1], n)``

    Returns
    -------
    lub : :obj:`numpy.ndarray` of shape (2 * l_and_u[0] + 1 + l_and_u[1], n)
        The LU decomposition of ``ab`` in banded storage format (see Notes).
        Its main diagonal is implicitly taken to be a vector of ones.
    ipiv : :obj:`numpy.ndarray` of shape (n,)
        The pivoting indices that define the permutation matrix ``P``.
    l_and_u : (:obj:`int`, :obj:`int`)
        The number of sub- and super-diagonals of ``ab`` that are non-zero.
    singular : :obj:`bool`
        A flag indicating whether the matrix is singular (``True``) or not (``False``).

    Raises
    ------
    TypeError
        If ``l_and_u`` or ``ab`` are not of the expected type.
    ValueError
        If ``l_and_u`` does not contain 2 non-negative integers.
    ValueError
        If the shape of ``ab`` is not compatible with the number of sub- and
        super-diagonals specified in ``l_and_u`` and ``ab_has_added_workspace``.

    Notes
    -----
    For LAPACK's banded LU decomposition, the matrix ``a`` is stored in ``ab`` using the
    matrix diagonal ordered form:

    ```python
    ab[u + i - j, j] == a[i,j] # see below for u
    ```

    An example of ``ab`` (shape of a is ``(7,7)``, ``u = 3`` superdiagonals, ``l = 2``
    subdiagonals) looks like

    ```python
    ab = np.array(
        [
            [   x,   x,   x, a03, a14, a25, a36],
            [   x,   x, a02, a13, a24, a35, a46],
            [   x, a01, a12, a23, a34, a45, a56],  # ↑ super-diagonals
            [ a00, a11, a22, a33, a44, a55, a66],  # main diagonal
            [ a10, a21, a32, a43, a54, a65,   x],  # ↓ sub-diagonals
            [ a20, a31, a42, a53, a64,   x,   x],
            [ a30, a41, a52, a63,   x,   x,   x],
        ]
    )
    ```

    where all entries marked with `x` are zero elements although they will be set to
    arbitrary values by this function.

    Internally LAPACK relies on an expanded version of this format to perform inplace
    operations that adds another ``l`` superdiagonals to the matrix in order to
    overwrite them for the purpose of pivoting, i.e.,

    ```python
    ab = np.array(
        [
            [   x,   x,   x,   x,   x,   x,   x],  # additional workspace
            [   x,   x,   x,   x,   x,   x,   x],  # additional workspace
            [   x,   x,   x, a03, a14, a25, a36],
            [   x,   x, a02, a13, a24, a35, a46],
            [   x, a01, a12, a23, a34, a45, a56],  # ↑ super-diagonals
            [ a00, a11, a22, a33, a44, a55, a66],  # main diagonal
            [ a10, a21, a32, a43, a54, a65,   x],  # ↓ sub-diagonals
            [ a20, a31, a42, a53, a64,   x,   x],
            [ a30, a41, a52, a63,   x,   x,   x],
        ]
    )
    ```

    This matrix is factorized into the upper triangular matrix ``U`` and the unit
    lower triangular matrix ``L`` as

    ```python
    lub = np.array(
        [
            [   x,   x,   x,   x,   x, u05, u16],
            [   x,   x,   x,   x, u04, u15, u26],
            [   x,   x,   x, u03, u14, u25, u36],
            [   x,   x, u02, u13, u24, u35, u46],
            [   x, u01, u12, u23, u34, u45, u56],  #
            [ u00, u11, u22, u33, u44, u55, u66],  # ↑ matrix U
            [ l10, l21, l32, l43, l54, l65,   x],  # ↓ matrix L
            [ l20, l31, l42, l53, l64,   x,   x],
            [ l30, l41, l52, l63,   x,   x,   x],
        ]
    )
    ```

    The main diagonal of ``L`` is implicitly taken to be a vector of ones.

    """  # noqa: E501

    # === Input validation ===

    # first, the number of sub- and super-diagonals needs to be validated
    if not isinstance(l_and_u, (tuple, list)):
        # NOTE: # removes the "<class '" and "'>" parts
        l_and_u_type_name = f"{type(l_and_u)}"[7:-1]
        raise TypeError(
            f"Expected 'l_and_u' to be a 2-tuple or 2-list of non-negative integers, "
            f"but got an object of type {l_and_u_type_name}."
        )

    l_and_u = tuple(  # type: ignore
        get_validated_integer(
            value=value,
            name=f"l_and_u[{index}]",
            min_value=0,
            max_value=None,
        )
        for index, value in enumerate(l_and_u)
    )

    if len(l_and_u) != 2:
        raise ValueError(
            f"Expected 'l_and_u' to be a tuple of 2 non-negative integers, but got "
            f"an object of length {len(l_and_u)}."
        )

    # then, the number of lower and upper subdiagonals needs to be checked for being
    # consistent with the shape of ``ab``
    num_subdiagonals, num_superdiagonals = l_and_u
    required_num_rows = (
        (1 + int(ab_has_added_workspace)) * num_subdiagonals + 1 + num_superdiagonals
    )
    min_num_columns = max(num_subdiagonals, num_superdiagonals) + 1

    ab_internal = get_validated_numeric_nd_array_like(
        value=ab,
        name="ab",
        dim=2,
        shape_limits=[(required_num_rows, required_num_rows), (min_num_columns, None)],
        dtype_kind=NumPyDTypeKinds.FLOAT_OR_COMPLEX,
        enforce_finite=check_finite,
        output_dtype=None,
    )

    # === Computation ===

    # if it was not done before, the additional workspace for pivoting needs to be added
    if not ab_has_added_workspace:
        # NOTE: a new Array will now be allocated, so it can be safely overwritten
        overwrite_ab = True
        ab = np.concatenate(
            (
                np.empty(
                    shape=(num_subdiagonals, ab_internal.shape[1]),
                    dtype=ab_internal.dtype,
                ),
                ab_internal,
            ),
            axis=0,
        )

    # now, the LAPACK-routines can be called
    # FIXME: for tridiagonal matrices, the SciPy wrapper for ``gttrf`` should be used
    lapack_routine = "gbtrf"
    (gbtrf,) = lapack.get_lapack_funcs((lapack_routine,), (ab,))

    lub, ipiv, info = gbtrf(
        ab=ab_internal,
        kl=num_subdiagonals,
        ku=num_superdiagonals,
        overwrite_ab=overwrite_ab,
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
        f"Illegal value in {-info}-th argument of internal {lapack_routine}."
    )


def lu_solve_banded(
    lub_factorization: BandedLUFactorization,
    b: ArrayLike,
    *,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> np.ndarray:
    """
    Solves a linear system of equations ``Ax=b`` with a banded matrix ``A`` using its
    precomputed LU decomposition.
    This function wraps the LAPACK-routine ``?gbtrs`` in an analogous way to SciPy's
    :func:`scipy.linalg.cho_solve_banded`.

    Parameters
    ----------
    lub_factorization : :obj:`BandedLUFactorization`
        The LU decomposition of the matrix ``A`` in banded storage format as returned by
        the function :func:`lu_banded`.
    b : :obj:`numpy.ndarray` of shape (n,) or (n, m)
        A right-hand side vector or matrix.
        Multiple right-hand sides are stored column-wise.
    overwrite_b : :obj:`bool`, default=``False``
        Whether the ``b`` input is allowed to be overwritten by the routine (``True``)
        or not (``False``).
    check_finite : :obj:`bool`, default=``True``
        Whether to check that ``b`` contains only finite numbers. Disabling may give a
        performance gain, but may result in problems (crashes, non-termination) if the
        inputs do contain infinities or NaNs.

    Returns
    -------
    x : ndarray of shape (n,) or (n, m)
        The solution(s) to the system ``A x = b``.
        Multiple solutions are stored column-wise.

    Raises
    ------
    TypeError
        If ``lub_factorization`` or ``b`` are not of the expected type.
    ValueError
        If the number of rows of ``b`` is not equal to the number of columns of the
        matrix ``A``.
    LinAlgError
        If the system to solve is singular.

    """

    # === Input Validation ===

    # if the matrix is singular, the solution cannot be computed
    if lub_factorization.singular:
        raise np.linalg.LinAlgError("Matrix is singular.")

    # then, it is attempted to convert ``b`` to a validated 1D-Array
    try:
        b_internal = get_validated_numeric_nd_array_like(
            value=b,
            name="b",
            dim=1,
            shape_limits=[(lub_factorization.lub.shape[1], None)],
            dtype_kind=NumPyDTypeKinds.FLOAT_OR_COMPLEX,
            enforce_finite=check_finite,
            output_dtype=None,
        )

    # if the 1D-conversion fails, it is attempted to convert ``b`` to a validated
    # 2D-Array
    except ValueError:
        b_internal = get_validated_numeric_nd_array_like(
            value=b,
            name="b",
            dim=2,
            shape_limits=[(lub_factorization.lub.shape[1], None), (1, None)],
            dtype_kind=NumPyDTypeKinds.FLOAT_OR_COMPLEX,
            enforce_finite=check_finite,
            output_dtype=None,
        )

    # === Computation ===

    # if b is not a view but a copy, it is allowed to be overwritten independently of
    # the user's choice
    overwrite_b = overwrite_b or not _is_data_linked(arr=b_internal, original=b)

    # now, the LAPACK-routine is called
    (gbtrs,) = lapack.get_lapack_funcs(("gbtrs",), (lub_factorization.lub, b_internal))
    x, info = gbtrs(
        ab=lub_factorization.lub,
        kl=lub_factorization.l_and_u[0],
        ku=lub_factorization.l_and_u[1],
        b=b_internal,
        ipiv=lub_factorization.ipiv,
        overwrite_b=overwrite_b,
    )

    # then, the results needs to be validated and returned
    # Case 1: the solution could be computed truly successfully, i.e., without any
    # NaN-values
    if info == 0 and not np.isnan(x).any():
        return x

    # Case 2: the solution was computed, but there were NaN-values in it
    elif info == 0:
        raise np.linalg.LinAlgError("Matrix is singular.")

    # Case 3: the solution could not be computed due to invalid input
    elif info < 0:  # pragma: no cover
        raise ValueError(f"Illegal value in {-info}-th argument of internal gbtrs.")

    # Case 4: unexpected error
    raise AssertionError(  # pragma: no cover
        f"The internal gbtrs returned info > 0 ({info}) which should not happen."
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
    lub_factorization : :obj:`BandedLUFactorization`
        The LU decomposition of the matrix ``A`` in banded storage format as returned by
        the function :func:`lu_banded`.

    Returns
    -------
    sign : :obj:`float`
        A number representing the sign of the determinant.
    logabsdet : :obj:`float`
        The natural log of the absolute value of the determinant.
        If the determinant is zero, then `sign` will be ``0`` and `logabsdet` will be
        ``-inf``. In all cases, the determinant is equal to
        ``sign * np.exp(logabsdet)``.

    Raises
    ------
    OverflowError
        If any of the diagonal entries of the LU decomposition leads to an overflow in
        the natural logarithm.

    """

    # first, the number of actual row exchanges needs to be counted
    # NOTE: ``ipiv[i]`` gives the index of the row which was used to pivot with row
    #       ``i``, so an exchange happened whenever ``ipiv[i] != i``
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
    # product of L and the diagonal product of U, the calculation simplifies
    # NOTE: as the main diagonal of L is a vector of ones, only the diagonal product of
    #       U is required
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
