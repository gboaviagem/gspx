"""Utilities for array manipulations."""
import numpy as np


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
    idx = np.where(M != 0)
    for xi, yi in zip(idx[0], idx[1]):
        q = M[xi, yi]
        mat[xi, yi, 1:] = q.vector
        mat[xi, yi, 0] = q.scalar

    return mat


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
