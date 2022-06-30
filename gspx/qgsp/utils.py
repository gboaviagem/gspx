"""Utilities."""
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
from gspx.qgsp.qgsp import QMatrix


def create_quaternion_weights(A, df, icols, jcols, kcols):
    """Populate a weighted adjacency matrix with quaternions.

    It is assumed that `A` is a weighted adjacency matrix
    and the features in `df` are assigned, row by row, to vertices
    in the graph.

    Not very optimized!

    """
    idx_nz = np.where(A != 0)
    entries = A[idx_nz]
    shape = A.shape

    cols = icols + jcols + kcols
    x = idx_nz[0]
    y = idx_nz[1]
    entriesq = []
    df_ = df[cols]
    for i, entry in enumerate(tqdm(entries)):
        diff = (df_.iloc[x[i], :] - df_.iloc[y[i], :]).abs()
        q = Quaternion(
            entry,
            np.linalg.norm(diff[icols]),
            np.linalg.norm(diff[jcols]),
            np.linalg.norm(diff[kcols])
        )
        entriesq.append(q)

    Aq = QMatrix.from_sparse(np.array(entriesq), idx_nz=idx_nz, shape=shape)
    return Aq
