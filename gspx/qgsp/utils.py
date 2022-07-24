"""Utilities."""
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
from gspx.qgsp.qgsp import QMatrix


def create_quaternion_weights(A, df, icols, jcols, kcols, gauss_den=10):
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
    for i, entry in enumerate(tqdm(entries)):
        diff = (df_.iloc[x[i], :] - df_.iloc[y[i], :]).abs()
        q = Quaternion(
            entry,
            np.linalg.norm(diff[icols]),
            np.linalg.norm(diff[jcols]),
            np.linalg.norm(diff[kcols])
        )
        entriesq.append(Quaternion.exp(q / gauss_den).inverse)

    Aq = QMatrix.from_sparse(np.array(entriesq), idx_nz=idx_nz, shape=shape)
    return Aq
