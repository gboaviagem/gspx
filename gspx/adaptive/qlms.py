"""QLMS algorithm."""
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from gspx.signals import QuaternionSignal

from gspx.utils.quaternion_matrix import explode_quaternions, \
    implode_quaternions
from gspx.qgsp import QMatrix


class QLMS:
    """Implementation of the QLMS algorithm."""

    def __init__(
            self, max_iter=100, alpha=None, scale=True,
            early_stopping_patience=10):
        """Construct."""
        if alpha is None:
            alpha = [0.01, 0.03, 0.1, 0.3, 1, 1.3]
        if not isinstance(alpha, list):
            alpha = [alpha]

        self.max_iter = max_iter
        self.alpha = alpha
        self.scale = scale
        self.early_stopping_patience = early_stopping_patience

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
        for layer in range(dims):
            new_column = (
                np.ones(rows)[:, np.newaxis] if layer == 0
                else np.zeros(rows)[:, np.newaxis])
            layers.append(
                np.hstack((
                    new_column,
                    arr[:, :, layer]
                ))
            )

        Xq = implode_quaternions(np.dstack(layers))
        return QMatrix.from_matrix(Xq)

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
            return QuaternionSignal.from_equal_dimensions(
                np.zeros((length, 1)))

        raise NotImplementedError(
            f"The option {method} is not available.")

    def fit(self, X, y):
        """Run the LMS algorithm."""
        m, d = X.shape
        X_ = QLMS.add_intercept_term(X)
        if self.scale:
            X_, mu, std = QLMS.normal_scaling(X_, ignore_first=True)
            self.mu_ = mu
            self.std_ = std

        res = {}
        for lr in self.alpha:
            theta = QLMS.initiate(length=d + 1, method='zeros')
            J = []

            for _ in tqdm(range(self.max_iter), desc=f"LR: {lr}"):
                err = X_ * theta - y

                # Calculate the J term, which is the current MSE
                cost = err.transpose() * err * (0.5/m)
                J.append(np.abs(cost.matrix.ravel()[0]))
                if len(J) > self.early_stopping_patience:
                    if (np.diff(J[-self.early_stopping_patience:]) > 0).all():
                        print(
                            "Early stopping due to increasing cost in "
                            f"the last {self.early_stopping_patience} "
                            "iterations.")
                        break

                # The gradient
                grad = X_.transpose().conjugate() * err * (1/m)

                # Here is the actual update
                theta = grad * (-1 * lr) + theta
            res[lr] = dict(result=theta, cost=J)

        self.res_ = res
        idx_lower_cost = np.argmin([
            np.abs(v['cost'][-1]) for _, v in res.items()])
        self.best_lr_ = self.alpha[idx_lower_cost]

        return self

    def predict(self, X):
        """Multiply the feature matrix with the optimum vector."""
        X_ = QLMS.add_intercept_term(X)
        if self.scale:
            X_.matrix[:, 1:] = (
                X_[:, 1:] - self.mu_[1:]).matrix / self.std_[1:]
        best_theta = self.res_[self.best_lr_]['result']
        return X_ * best_theta

    def plot(self, prune_early_stopped=True, nsamples=50):
        """Show the cost function per iteration."""
        plotstyle = ['b', 'r', 'g', 'k', 'b--', 'r--']

        plt.figure()
        for i, val in enumerate(self.res_.keys()):
            length = len(self.res_[val]['cost'])
            if prune_early_stopped:
                if length < self.max_iter:
                    continue
            length = min(nsamples, length)
            plt.plot(
                np.arange(self.max_iter)[:length],
                self.res_[val]['cost'][:length],
                plotstyle[i],
                label=val)
        plt.ylabel("Cost function (MSE)")
        plt.legend(loc="upper right")

        plt.show()
