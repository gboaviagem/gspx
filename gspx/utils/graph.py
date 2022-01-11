"""Utilities for creating or modifying graphs matrices."""

import numpy as np
from scipy.sparse import csr_matrix, kron


def adj_matrix_path(N, weights=None, directed=False):
    """Return the adjacency matrix of a path graph.

    Parameters
    ----------
    N: int
            Number of graph nodes.
    weights: array, default=None
            Array with edge weights. If None, unit weights are used.
            If not None, then the given value of N is replaced by
            weights + 1.
    directed: bool, default=False
            If True, a directed graph is created.

    Returns
    -------
    A : np.ndarray, shape=(N,N)

    """
    if weights is None:
        A = np.tri(N, k=1) - np.tri(N, k=-2) - np.eye(N)
    else:
        N = len(weights) + 1
        A = np.zeros((N, N))
        A[:-1, 1:] = np.diag(weights)
        A = A + A.transpose()
    if directed:
        A = np.tril(A)
    return A


def coords_path(N):
    """Coordinates of the vertices in the path graph.

    Parameters
    ----------
    N : int
            Number of graph vertices.

    Returns
    -------
    coords : np.ndarray, shape=(N,2)

    """
    coords = np.array([[i, 0] for i in range(N)])
    return coords


def make_path(N, weights=None, directed=False):
    """Create adjacency matrix and coordinates of a path graph.

    Parameters
    ----------
    N: int
            Number of graph nodes.
    weights: array, default=None
            Array with edge weights. If None, unit weights are used.
            If not None, then the given value of N is replaced by
            weights + 1.
    directed: bool, default=False
            If True, a directed graph is created.

    Returns
    -------
    A : np.ndarray, shape=(N,N)
    coords : np.ndarray, shape=(N,2)

    """
    if weights is not None:
        assert N == len(weights) + 1, (
            "Length of weights array is {}, not compatible with "
            "{} vertices.".format(len(weights), N)
        )
    A = adj_matrix_path(N, weights=weights, directed=directed)
    coords = coords_path(N)
    return A, coords


def make_grid(rows, columns, weights_r=None, weights_c=None):
    """Create a grid graph.

    By "grid graph" we mean the underlying domain of digital images,
    as usually modelled by a graph in which each pixel rests on a node
    and each node is only connected to its direct neighbors, in the
    vertical and horizontal directions.

    Parameters
    ----------
    rows: int
            Number of rows in the grid.
    columns: int
            Number of columns in the grid.
    weights_r: array, default=None
            Weights in the rows. If None, unit weights are considered.
    weights_c: array, default=None
            Weights in the columns. If None, unit weights are considered.

    Returns
    -------
    A : scipy.sparse.csr_matrix, shape=(N,N)
    coords : np.ndarray, shape=(N,2)

    """
    A1, coords1 = make_path(columns, weights=weights_c)
    A2, coords2 = make_path(rows, weights=weights_r)

    N1 = len(A1)
    N2 = len(A2)

    # Using the property that the grid graph is the cartesian product
    # of two path graphs.
    A = kron(
        csr_matrix(A1), csr_matrix(np.eye(N2))
    ) + kron(
        csr_matrix(np.eye(N1)), csr_matrix(A2)
    )
    coords = list()
    for c1 in coords1[:, 0].ravel():
        for c2 in coords2[:, 0].ravel():
            coords.append([c1, c2])
    coords = np.array(coords)

    return A, coords
