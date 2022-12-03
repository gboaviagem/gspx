"""QLMS algorithm."""
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from gspx.signals import QuaternionSignal

from gspx.qgsp import QMatrix
from typing import Union, List


class QLMS:
    """Implementation of the QLMS algorithm.

    Parameters
    ----------
    max_iter: int = 50
    step_size: Union[float, List[float]] = None
        Learning rate, or step size, of the algorithm.
    scale: bool = True
    early_stopping_patience: int = 3
        The amount of iterations in which we tolerate the cost to grow
        successively. After that, the execution for the given
        step size
    verbose: Union[int, bool] = 0

    References
    ----------
    Took, Clive Cheong, and Danilo P. Mandic. "The quaternion LMS
    algorithm for adaptive filtering of hypercomplex processes."
    IEEE Transactions on Signal Processing 57.4 (2008): 1316-1327.

    """

    def __init__(
            self, max_iter: int = 100,
            step_size: Union[float, List[float]] = None,
            scale: bool = True,
            early_stopping_patience: int = 3,
            verbose: Union[int, bool] = 0):
        """Construct."""
        if isinstance(verbose, bool):
            verbose = int(verbose)
        self.verbose = verbose

        if step_size is None:
            step_size = [0.01, 0.03, 0.1, 0.3, 1, 1.3]
        if not isinstance(step_size, list):
            step_size = [step_size]

        self.max_iter = max_iter
        self.step_size = step_size
        self.scale = scale
        self.early_stopping_patience = early_stopping_patience

        self.mu_ = None
        self.std_ = None
        self.res_ = None
        self.best_lr_ = None

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

    def fit(self, X: QMatrix, y: QMatrix):
        """Run the QLMS algorithm."""
        m, d = X.shape
        X_ = X.copy()
        if self.scale:
            X_, mu, std = QLMS.normal_scaling(X_, ignore_first=True)
            self.mu_ = mu
            self.std_ = std

        res = {}
        for lr in self.step_size:
            theta = QLMS.initiate(length=d, method='zeros')
            J = []

            for _ in (
                    tqdm(range(self.max_iter), desc=f"LR: {lr}")
                    if self.verbose > 0 else range(self.max_iter)
            ):
                err = y - (theta.transpose() * X_.transpose()).transpose()

                # Calculate the J term, which is the current MSE
                cost = err.transpose() * err.conjugate() * (0.5/m)
                J.append(np.abs(cost.matrix.ravel()[0]))
                if len(J) > self.early_stopping_patience:
                    if (np.diff(J[-self.early_stopping_patience:]) > 0).all():
                        print(
                            "Early stopping due to increasing cost in "
                            f"the last {self.early_stopping_patience} "
                            "iterations.")
                        break

                # The gradient
                grad = (
                    (err.transpose() * X_.conjugate()).transpose() * 2 -
                    X_.conjugate().transpose() * err.conjugate()
                )

                # Here is the actual update
                # theta = grad * (-1 * lr) + theta
                theta = theta + grad * lr

            if self.verbose > 1:
                print("Cost per iteration:", J)

            res[lr] = dict(result=theta, cost=J)

        self.res_ = res
        idx_lower_cost = np.argmin([
            np.abs(v['cost'][-1]) for _, v in res.items()])
        self.best_lr_ = self.step_size[idx_lower_cost]

        return self

    def predict(self, X: QMatrix):
        """Multiply the feature matrix with the optimum vector."""
        X_ = X.copy()
        if self.scale:
            X_.matrix[:, 1:] = (
                X_[:, 1:] - self.mu_[1:]).matrix / self.std_[1:]
        best_theta = self.res_[self.best_lr_]['result']
        return (best_theta.transpose() * X_.transpose()).transpose()

    def plot(self, prune_early_stopped: bool = True, nsamples: int = 50):
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
