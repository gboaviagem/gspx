"""Utilities."""
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
from gspx.qgsp.qgsp import QMatrix
import pandas as pd
from typing import Union


def create_quaternion_weights(
        A: np.ndarray, df: pd.DataFrame,
        icols: list, jcols: list, kcols: list,
        gauss_den: Union[int, float] = 10) -> QMatrix:
    """Populate a weighted adjacency matrix with quaternions.

    It is assumed that `A` is a weighted adjacency matrix
    and the features in `df` are assigned, row by row, to vertices
    in the graph.

    Not very optimized!

    Parameters
    ----------
    A : np.ndarray
        Real-valued adjacency matrix.
    df : pd.DataFrame
        Dataframe with data used to compose the i, j and k weight
        dimensions.
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

    """
    idx_nz = np.where(A != 0)
    entries = np.array(A[idx_nz]).ravel()
    shape = A.shape

    cols = icols + jcols + kcols
    x = idx_nz[0]
    y = idx_nz[1]
    entriesq = []
    df_ = df[cols]
    remove_idx = []
    for i, entry in enumerate(tqdm(entries)):
        diff = (df_.iloc[x[i], :] - df_.iloc[y[i], :]).abs()
        q = Quaternion(
            entry,
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

    Aq = QMatrix.from_sparse(np.array(entriesq), idx_nz=idx_nz, shape=shape)
    return Aq
