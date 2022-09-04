"""Utilities for creating or modifying graphs matrices."""

import numpy as np
from scipy.sparse import csr_matrix, kron
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import networkx as nx


def adj_matrix_ring(N=None, weights=None):
    """Return the adjacency matrix of a path graph.

    Parameters
    ----------
    N: int, default=None
        Number of graph nodes.
    weights: array, default=None
        Array with edge weights.

    Returns
    -------
    A : np.ndarray, shape=(N,N)

    """
    assert N is not None or weights is not None, (
        "Either 'N' or 'weights' must be given."
    )
    if N is not None and weights is not None:
        print("Ignoring 'N' since 'weights' was also given.")

    if weights is None:
        weights = np.zeros(N) + 1

    return np.roll(np.diag(weights), shift=1, axis=1)


def coords_ring_graph(N):
    """Return the vertices coordinates of a ring graph.

    Parameters
    ----------
    N: int
        Number of graph nodes.

    """
    coords = np.zeros((N, 2))
    n = np.arange(N)
    coords[:, 0] = np.cos(2.0*np.pi*n/N)
    coords[:, 1] = -np.sin(2.0*np.pi*n/N)
    return coords


def adj_matrix_path(N=None, weights=None, directed=False):
    """Return the adjacency matrix of a path graph.

    Parameters
    ----------
    N: int, default=None
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
    assert N is not None or weights is not None, (
        "Either 'N' or 'weights' must be given."
    )
    if N is not None and weights is not None:
        print("Ignoring 'N' since 'weights' was also given.")

    if weights is None:
        A = np.tri(N, k=1) - np.tri(N, k=-2) - np.eye(N)
    else:
        assert isinstance(weights, np.ndarray)
        N = len(weights) + 1
        A = np.zeros((N, N), dtype=weights.dtype)
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


def make_sensor(N, n_neighbors=5, seed=42, directed=False):
    """Make a sensor graph.

    Parameters
    ----------
    N : int
        Number of nodes.
    n_neighbors : int, default=5
    seed : int, default=42
    directed : bool, default=False

    Example
    -------
    >>> A, coords = make_sensor(10, n_neighbors=5, directed=True)
    >>> A.sum(axis=1)
    array([1.21202674, 0.72507085, 2.07115662, 0.77018975, 2.02283878,
           1.52266368, 0.92886763, 1.5342546 , 1.83783597, 0.70598149])
    >>> coords
    array([[0.77395605, 0.43887844],
           [0.85859792, 0.69736803],
           [0.09417735, 0.97562235],
           [0.7611397 , 0.78606431],
           [0.12811363, 0.45038594],
           [0.37079802, 0.92676499],
           [0.64386512, 0.82276161],
           [0.4434142 , 0.22723872],
           [0.55458479, 0.06381726],
           [0.82763117, 0.6316644 ]])

    """
    rnd = np.random.default_rng(seed)
    coords = rnd.uniform(low=0.0, high=1.0, size=(N, 2))
    A = nearest_neighbors(coords, n_neighbors=n_neighbors)
    A = A.toarray()
    if not directed:
        A = A + A.T
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


def nearest_neighbors(
        X, n_neighbors=20, algorithm='ball_tree', mode='distance'):
    """Return the nearest neighbors' graph weighted adjacency matrix.

    This is a wrapper for the Scikit-learn NearestNeighbors.kneighbors_graph
    method.

    Parameters
    ----------
    X : np.ndarray()
        Array of features.
    n_neighbors : int, optional, default: 20
    algorithm : str, optional, default: 'ball_tree'
    mode : str, optional, default: 'distance'

    Return
    ------
    W : weighted adjacency matrix in CSR (Compressed Sparse Row) format

    """
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm=algorithm).fit(X)
    W = nbrs.kneighbors_graph(X, mode=mode)

    return W


def adj_matrix_from_coords(coords, theta, verbose=False):
    """Create a gaussian-weighted adjacency matrix using euclidean distance.

    Nodes for which the distance is greater than 2*theta are ignored.

    Parameters
    ----------
    coords : array
        (N, 2) array of coordinates.
    theta : float
        Variance of the weight distribution.

    """
    [N, M] = coords.shape
    A = np.zeros((N, N))
    for i in (tqdm(np.arange(1, N)) if verbose else np.arange(1, N)):
        for j in np.arange(i):
            x1 = coords[i, 0]
            y1 = coords[i, 1]
            x2 = coords[j, 0]
            y2 = coords[j, 1]
            distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if distance < 2 * theta:
                A[i, j] = np.exp(-(distance ** 2)/(2 * theta ** 2))
    if verbose:
        print("adj_matrix_from_coords process is completed.")
    return A + A.transpose()


def adj_matrix_from_coords_limited(coords, limit, theta=1, verbose=False):
    """Create a nearest-neighbors graph with gaussian weights.

    Parameters
    ----------
    coords : array
        (N, 2) array of coordinates.
    limit : int
        Minimum number of neighbors.
    theta : float
        Variance of the gaussian weight distribution.

    """
    N, _ = coords.shape
    A = np.zeros((N, N))
    for i in (tqdm(np.arange(1, N)) if verbose else np.arange(1, N)):
        dist2i = np.sqrt(
            (coords[:, 0] - coords[i, 0]) ** 2 +
            (coords[:, 1] - coords[i, 1]) ** 2)

        idx = np.argsort(dist2i)[1: limit + 1]
        for j in idx:
            x1 = coords[i, 0]
            y1 = coords[i, 1]
            x2 = coords[j, 0]
            y2 = coords[j, 1]
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if A[i, j] == 0:
                A[i, j] = np.exp(-(distance ** 2)/(2 * theta ** 2))

    return A + A.transpose()


def to_networkx(A):
    """Create a Networkx graph out of a weighted adjacency matrix.

    Parameters
    ----------
    A : CSR matrix or np.ndarray, shape=(N, N)

    Returns
    -------
    g : nx.Graph or nx.DiGraph
        A directed or undirected graph, depending on whether A
        is symmetric.

    """
    if isinstance(A, csr_matrix):
        A_ = A.toarray()
    else:
        A_ = A.copy()
    is_symmetric = np.allclose(A, A.T, rtol=1e-05, atol=1e-08)

    xi, yi = np.where(A_)
    edgelist = [
        (xi[n], yi[n], {'weight': A_[xi[n], yi[n]]})
        for n in range(len(xi))
    ]

    if is_symmetric:
        g = nx.Graph()
    else:
        g = nx.DiGraph()
    g.add_edges_from(edgelist)

    return g


def graph_total_variation(X, S, tv_norm=2):
    """Compute the graph total variation of column vectors.

    Parameters
    ----------
    X : np.ndarray, shape=(N, K)
        K column vectors.
    S : np.ndarray, shape=(N, N)
        Graph shift operator.
    tv_norm : int, default=2
        Order of the norm computed inside the total variation.

    Returns
    -------
    tv : np.ndarray, shape=(K,)
        Total variation of all the column vectors.

    Notes
    -----
    The parameter `tv_norm` is passed as the `ord` parameters to
    numpy.linalg.norm() [1]_.

    References
    ----------
    [1] https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html

    """
    X_shifted = S @ X
    return np.linalg.norm(X_shifted - X, axis=0, ord=tv_norm)


def describe(graph, return_dict=False):
    """Compute and print some graph-related metrics.

    Parameters
    ----------
    graph : nx.Graph() or np.ndarray
        If Numpy array, it is assumed to be the adjacency matrix.
    return_dict : bool, optional, default: False
        Whether to return a dictionary with info.

    """
    if isinstance(graph, np.ndarray):
        graph = nx.from_numpy_matrix(graph)

    if isinstance(graph, nx.DiGraph):  # MultiDiGraph is also considered here
        is_dir = True
    else:
        is_dir = False

    d = dict()
    d["n_nodes"] = nx.number_of_nodes(graph)
    d["n_edges"] = nx.number_of_edges(graph)
    d["n_self_loops"] = nx.number_of_selfloops(graph)
    d["density"] = nx.density(graph)
    if is_dir:
        d["is_strongly_connected"] = nx.is_strongly_connected(graph)
        d["n_strongly_connected_components"] = \
            nx.number_strongly_connected_components(graph)
        d["is_weakly_connected"] = nx.is_weakly_connected(graph)
        d["n_weakly_connected_components"] = \
            nx.number_strongly_connected_components(graph)
    else:
        d["is_connected"] = nx.is_connected(graph)
        d["n_connected_components"] = nx.number_connected_components(graph)
    d["is_directed"] = nx.is_directed(graph)
    d["is_weighted"] = nx.is_weighted(graph)

    if return_dict:
        return d
    else:
        for key, value in d.items():
            print(key + ":", value)
