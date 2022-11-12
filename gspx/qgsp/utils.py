"""Utilities."""
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
from gspx.qgsp.qgsp import QMatrix
import pandas as pd
from typing import List, Union


def create_quaternion_weights(
        A: np.ndarray, df: pd.DataFrame,
        cols1: List[str], icols: List[str],
        jcols: List[str], kcols: List[str],
        gauss_den: Union[int, float] = 10,
        hermitian: bool = True,
        sparse_output: bool = False,
        verbose: bool = True) -> Union[QMatrix, tuple]:
    """Populate a weighted adjacency matrix with quaternions.

    It is assumed that `A` is an adjacency matrix
    and the features in `df` are assigned, row by row, to vertices
    in the graph. `A` is provided only to capture connectivity, the values
    in its entries do not matter, as long as they are different than zero.

    Not very optimized.

    Parameters
    ----------
    A : np.ndarray
        Real-valued adjacency matrix.
    df : pd.DataFrame
        Dataframe with data used to compose the i, j and k weight
        dimensions.
    cols1 : list of str
        Columns whose distance will compose the real part.
    icols : list of str
        Columns whose distance will compose the i dimension.
    jcols : list of str
        Columns whose distance will compose the j dimension.
    kcols : list of str
        Columns whose distance will compose the k dimension.
    gauss_den : int, default=10
        Integer assigned to the denominator in the gaussian
        weight distribution, as in `exp(- (x) / gauss_den)`.
        It is related to the gaussian standard deviation.
    hermitian : bool, default=True
        If True, make the quaternion output matrix hermitian.
    sparse_output : bool, default=True
        If True, only the sparse data (indices and entries) are returned.
    verbose : bool, default = True

    Return
    ------
    If `sparse_output` is True, it returns the tuple
    `(entriesq, idx_nz, shape)`, with the quaternion-valued non-zero entries.
    Otherwise, it returns the dense QMatrix object.

    """
    if hermitian:
        idx_nz = np.where(np.triu(A, k=1) != 0)
    else:
        idx_nz = np.where(A != 0)

    shape = A.shape

    cols = list(set(cols1 + icols + jcols + kcols))
    x = idx_nz[0]
    y = idx_nz[1]
    entriesq = []
    df_ = df[cols]
    remove_idx = []
    for i, xi in enumerate(tqdm(x) if verbose else x):
        diff = (df_.iloc[xi, :] - df_.iloc[y[i], :]).abs()
        q = Quaternion(
            np.linalg.norm(diff[cols1]),
            np.linalg.norm(diff[icols]),
            np.linalg.norm(diff[jcols]),
            np.linalg.norm(diff[kcols])
        )
        exp_ = Quaternion.exp(q / gauss_den)
        if exp_.norm > 1e-7:
            entriesq.append(exp_.inverse)
        else:
            remove_idx.append(i)

    idx_nz = (
        np.delete(idx_nz[0], remove_idx),
        np.delete(idx_nz[1], remove_idx)
    )

    if hermitian:
        entriesq = entriesq + [q.conjugate for q in entriesq]
        idx_nz = (
            np.array(idx_nz[0].tolist() + idx_nz[1].tolist()),
            np.array(idx_nz[1].tolist() + idx_nz[0].tolist())
        )

    if sparse_output:
        return np.array(entriesq), idx_nz, shape
    else:
        if verbose:
            print(
                "Please wait while the dense "
                "quaternion matrix is assembled.")
        return QMatrix.from_sparse(
            np.array(entriesq), idx_nz=idx_nz, shape=shape)
