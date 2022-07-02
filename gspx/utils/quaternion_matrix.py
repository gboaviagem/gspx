"""Utilities for array manipulations."""
import numpy as np
from pyquaternion import Quaternion


def _quaternion_mtx_assertions(M):
    """Assert the appropriate properties of the input."""
    assert isinstance(M.dtype, type(np.dtype('O'))) and len(M.shape) == 2


def explode_quaternions(M):
    """Turn a 2D quaternion array into a 3D real array.

    Parameters
    ----------
    M : np.array, shape=(n, m)
        Array of type np.dtype('O'). The non-zero entries are required
        to be "pyquaternion.Quaternion"-valued.

    Return
    ------
    mat : np.ndarray, shape=(n, m, 4)

    """
    _quaternion_mtx_assertions(M)

    # Here we assume M is a matrix in which the non-zero entries
    # are "pyquaternion.Quaternion"-valued
    n, m = M.shape
    mat = np.zeros((n, m, 4))
    idx = np.where(M != Quaternion(0, 0, 0, 0))
    for xi, yi in zip(idx[0], idx[1]):
        q = M[xi, yi]
        mat[xi, yi, 1:] = q.vector
        mat[xi, yi, 0] = q.scalar

    return mat


def implode_quaternions(M):
    """Turn a 3D real array into a 2D quaternion array.

    Parameters
    ----------
    M : np.ndarray, shape=(n, m, 4)

    Return
    ------
    mat : np.ndarray, shape=(n, 4)
        Array of type np.dtype('O'). The non-zero entries are
        "pyquaternion.Quaternion"-valued.

    """
    from pyquaternion import Quaternion
    mat = (
        M[:, :, 0] * Quaternion(1, 0, 0, 0) +
        M[:, :, 1] * Quaternion(0, 1, 0, 0) +
        M[:, :, 2] * Quaternion(0, 0, 1, 0) +
        M[:, :, 3] * Quaternion(0, 0, 0, 1)
    )
    return mat


def conjugate(M):
    """Conjugate of a quaternion-valued matrix.

    Parameters
    ----------
    M : np.ndarray, shape=(n, m)
        Array of type np.dtype('O'). The non-zero entries are required
        to be "pyquaternion.Quaternion"-valued.

    Return
    ------
    mat : np.ndarray, shape=(n, m)
        Array of type np.dtype('O'). The non-zero entries are required
        to be "pyquaternion.Quaternion"-valued.
        The imaginary part is the aditive inverse of that in `M`.

    """
    M_real = explode_quaternions(M)
    M_real[:, :, 1:] = - M_real[:, :, 1:]
    return implode_quaternions(M_real)


def symplectic_decompose_mtx(M):
    """Compute the symplectic decomposition of a quaternion matrix.

    Parameters
    ----------
    M : np.array, shape=(n, m)
        Array of type np.dtype('O'). The non-zero entries are required
        to be "pyquaternion.Quaternion"-valued.

    Return
    ------
    simp : complex-valued np.ndarray, shape=(n, m)
    perp : complex-valued np.ndarray, shape=(n, m)

    """
    M4 = explode_quaternions(M)
    simp = M4[:, :, 0] + 1j * M4[:, :, 1]
    perp = M4[:, :, 2] + 1j * M4[:, :, 3]
    return simp, perp


def complex_adjoint(M):
    """Compute the complex adjoint of a quaternion matrix.

    Parameters
    ----------
    M : np.array, shape=(n, m)
        Array of type np.dtype('O'). The non-zero entries are required
        to be "pyquaternion.Quaternion"-valued.

    Return
    ------
    mat : np.ndarray, shape=(2*n, 2*m)

    """
    simp, perp = symplectic_decompose_mtx(M)
    adj = np.concatenate((
        np.concatenate((simp, perp), axis=1),  # side-by-side concatenation
        np.concatenate((-perp.conjugate(), simp.conjugate()), axis=1)
    ), axis=0  # vertical concatenation
    )
    return adj


def from_exploded_to_complex(Q):
    """Get the complex representation of a quaternion matrix.

    The quaternion matrix is represented as a (N, M, 4) real-valued
    matrix. Otherwise, call `explode_quaternions()`.

    For each quaternion entry q = a + bi + cj + dk,
    the complex representation turns it into a 2x2 matrix
    [[a + bi, c + di], [-c + di, a - bi]].

    """
    nrows, ncols, _ = Q.shape

    null = np.zeros((2 * nrows, 2 * ncols), dtype=complex)

    simp = Q[:, :, 0] + 1j * Q[:, :, 1]
    perp = Q[:, :, 2] + 1j * Q[:, :, 3]

    c00 = null.copy()
    c00[0::2, 0::2] = simp
    c01 = null.copy()
    c01[0::2, 1::2] = perp
    c10 = null.copy()
    c10[1::2, 0::2] = - perp.conjugate()
    c11 = null.copy()
    c11[1::2, 1::2] = simp.conjugate()

    return c00 + c10 + c01 + c11


def from_complex_to_exploded(Q):
    """Get the exploded representation out of the complex form."""
    return np.dstack((
        np.real(Q[::2, ::2]),
        np.imag(Q[::2, ::2]),
        np.real(Q[::2, 1::2]),
        np.imag(Q[::2, 1::2])
    ))
