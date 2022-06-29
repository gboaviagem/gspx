"""QLMS algorithm."""
import numpy as np

from gspx.utils.quaternion_matrix import explode_quaternions
from gspx.qgsp import QMatrix


class QLMS:
    """Implementation of the QLMS algorithm."""

    def __init__(self, max_iter=100, alpha=None, scale=True):
        """Construct."""
        if alpha is None:
            alpha = [0.01, 0.03, 0.1, 0.3, 1, 1.3]
        if not isinstance(alpha, list):
            alpha = [alpha]

        self.max_iter = max_iter
        self.alpha = alpha
        self.scale = scale

        self.mu_ = None
        self.std_ = None
        self.res_ = None
        self.best_lr_ = None

    @staticmethod
    def add_intercept_term(X):
        """Add intercept in the feature matrix."""
        assert isinstance(X, QMatrix)
        arr = explode_quaternions(X.matrix)
        rows, _, dims = arr.shape

        layers = []
        for l in range(dims):
            layers.append(
                np.hstack((
                    np.ones(rows)[:, np.newaxis],
                    arr[:, :, l]
                ))
            )

        return np.dstack(layers)

    @staticmethod
    def normal_scaling(X, ignore_first=True):
        """Perform scaling in the feature matrix."""
        assert isinstance(X, QMatrix)
        mu = X.mean(axis=0)
        std = np.std(X.abs(), axis=0, ddof=1) + 1e-10
        X_ = X.copy()
        idx = 1 if ignore_first else 0
        X_.matrix[:, idx:] = (X[:, idx:] - mu[idx:]).matrix / std[idx:]
        return X_, mu, std

    @staticmethod
    def initiate(length, method='zeros'):
        """Initiate the array to be optimized."""
        opts = ['zeros']
        assert method in opts, (
            f"Invalid option, choose among {opts}.")

        if method == 'zeros':
            return np.zeros((length, 1))

        raise NotImplementedError(
            f"The option {method} is not available.")
