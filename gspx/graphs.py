"""Graph object."""

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, csgraph
from scipy.sparse.linalg import eigs
from gspx.base import Signal


class Graph:
    """Store and manipulate (ideally sparse) graphs.

    This relies heavily on the compressed-sparse graph structure:
    https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html

    Parameters
    ----------
    A : array-like
        Adjacency matrix. Shape should be (n_nodes, n_nodes).

    Attributes
    ----------
    adjacency : CSR-format sparse graph

    """

    def __init__(self, A):
        """Construct."""
        if isinstance(A, np.ndarray):
            self.adjacency = csr_matrix(A)
        elif isinstance(A, csr_matrix):
            self.adjacency = A
        else:
            raise ValueError("Please provide a numpy array or CSR matrix.")

    def compute_eigenbasis(
            self, k=6, M=None, sigma=None, return_eigenvectors=True,
            **kwargs):
        """Compute the Fourier eigenbasis.

        Parameters
        ----------
        k : int, optional
            The number of eigenvalues and eigenvectors desired.
            `k` must be smaller than N-1. It is not possible to compute all
            eigenvectors of a matrix.
        M : ndarray, sparse matrix or LinearOperator, optional
            An array, sparse matrix, or LinearOperator representing
            the operation M*x for the generalized eigenvalue problem
                A * x = w * M * x.
            M must represent a real symmetric matrix if A is real, and must
            represent a complex Hermitian matrix if A is complex. For best
            results, the data type of M should be the same as that of A.
            Additionally:
                If `sigma` is None, M is positive definite
                If sigma is specified, M is positive semi-definite
            If sigma is None, eigs requires an operator to compute the solution
            of the linear equation ``M * x = b``.
            This is done internally via a
            (sparse) LU decomposition for an explicit matrix M, or via an
            iterative solver for a general linear operator.  Alternatively,
            the user can supply the matrix or operator Minv, which gives
            ``x = Minv * b = M^-1 * b``.
        sigma : real or complex, optional
            Find eigenvalues near sigma using shift-invert mode.  This requires
            an operator to compute the solution of the linear system
            ``[A - sigma * M] * x = b``, where M is the identity matrix if
            unspecified. This is computed internally via a (sparse) LU
            decomposition for explicit matrices A & M, or via an iterative
            solver if either A or M is a general linear operator.
            Alternatively, the user can supply the matrix or operator OPinv,
            which gives ``x = OPinv * b = [A - sigma * M]^-1 * b``.
            For a real matrix A, shift-invert can either be done in imaginary
            mode or real mode, specified by the parameter OPpart ('r' or 'i').
            Note that when sigma is specified, the keyword 'which' (below)
            refers to the shifted eigenvalues ``w'[i]`` where:
                If A is real and OPpart == 'r' (default),
                ``w'[i] = 1/2 * [1/(w[i]-sigma) + 1/(w[i]-conj(sigma))]``.
                If A is real and OPpart == 'i',
                ``w'[i] = 1/2i * [1/(w[i]-sigma) - 1/(w[i]-conj(sigma))]``.
                If A is complex, ``w'[i] = 1/(w[i]-sigma)``.
        return_eigenvectors : bool, optional
            Return eigenvectors (True) in addition to eigenvalues
        **kwargs : dict
            Other keyword arguments. Please refer to the documentation
            on `scipy.sparse.linalg.eig`.

        Returns
        -------
        w : ndarray
            Array of k eigenvalues.
        v : ndarray
            An array of `k` eigenvectors.
            ``v[:, i]`` is the eigenvector corresponding to the
            eigenvalue w[i].

        """
        L = csgraph.laplacian(self.adjacency)
        # w : Array of k eigenvalues.
        # v : An array of k eigenvectors. v[:, i] is the eigenvector
        # corresponding to the eigenvalue w[i].
        w, v = eigs(
            L, k=k, M=M, sigma=sigma,
            return_eigenvectors=return_eigenvectors,
            **kwargs)

    def plot(self, color_signal=None, coords=None, alpha=0.7):
        """Display a representation of the graph.

        Parameters
        ----------
        colors : array-like, shape=(n_nodes, 3), default=None
        alpha : float, between 0 and 1, default=0.7
        coords : array-like, shape=(n_nodes, 3), default=None

        """
        if isinstance(color_signal, Signal):
            colors = color_signal.to_rgba()[:, :-1]  # ignore the alpha channel
        else:
            colors = color_signal

        kwargs = dict()
        if colors is not None:
            colors = [
                tuple(list(rgb) + [alpha])
                for rgb in colors / colors.max()
            ]
            kwargs['node_color'] = colors
        if coords is not None:
            kwargs['pos'] = coords

        G = nx.from_scipy_sparse_matrix(self.adjacency)
        nx.draw(G, **kwargs)
