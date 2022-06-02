"""Utils related to GSP."""

import numpy as np


def eigendecompose(S):
    """Eigendecompose the input matrix."""
    eigvals, V = np.linalg.eig(S)
    return eigvals, V


def gft(S, x, decompose_shift=True):
    """Perform the inverse GFT.

    Parameters
    ----------
    S : np.ndarray, shape=(N, N)
        Graph shift operator or GFT matrix.
    x : np.ndarray, shape=(N,) or (N, 1)
        Graph signal spectrum.
    decompose_shift : bool, default=True

    Example
    -------
    >>> from gspx.utils.gsp import gft, igft
    >>> from gspx.utils.graph import make_sensor
    >>> A, coords = make_sensor(N=10, seed=2)
    >>> s = np.arange(10)
    >>> s_ = gft(A, igft(A, s))
    >>> np.round(np.sum(s_ - s), decimals=10)
    0.0

    """
    if decompose_shift:
        _, U = eigendecompose(S)
        Uinv = np.linalg.inv(U)
    else:
        Uinv = S.copy()

    not_column = False
    if len(x.shape) == 1:
        x_column = x[:, np.newaxis]
        not_column = True
    else:
        x_column = x.copy()

    out = Uinv @ x_column

    if not_column:
        return out.ravel()
    else:
        return out


def igft(S, x, decompose_shift=True):
    """Perform the inverse GFT.

    Parameters
    ----------
    S : np.ndarray, shape=(N, N)
        Graph shift operator or IGFT matrix.
    x : np.ndarray, shape=(N,) or (N, 1)
        Graph signal spectrum.
    decompose_shift : bool, default=True

    Example
    -------
    >>> from gspx.utils.gsp import gft, igft
    >>> from gspx.utils.graph import make_sensor
    >>> A, coords = make_sensor(N=10, seed=2)
    >>> s = np.arange(10)
    >>> s_ = gft(A, igft(A, s))
    >>> np.round(np.sum(s_ - s), decimals=10)
    0.0

    """
    if decompose_shift:
        _, U = eigendecompose(S)
    else:
        U = S.copy()
    return gft(U, x, decompose_shift=False)
