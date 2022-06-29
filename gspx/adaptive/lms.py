"""LMS algorithm."""
import numpy as np


class LMS:
    """Implementation of the LMS algorithm."""

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
        m, _ = X.shape
        return np.hstack((np.ones((m, 1)), X))

    @staticmethod
    def normal_scaling(X, ignore_first=True):
        """Perform scaling in the feature matrix."""
        mu = X.mean(axis=0)
        std = np.std(X, axis=0, ddof=1) + 1e-10
        X_ = X.copy()
        idx = 1 if ignore_first else 0
        X_[:, idx:] = (X[:, idx:] - mu[idx:]) / std[idx:]
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

    def fit(self, X, y):
        """Run the LMS algorithm."""
        m, d = X.shape
        X_ = LMS.add_intercept_term(X)
        if self.scale:
            X_, mu, std = LMS.normal_scaling(X_, ignore_first=True)
            self.mu_ = mu
            self.std_ = std

        res = dict()
        for lr in self.alpha:
            theta = LMS.initiate(length=d + 1, method='zeros')
            J = np.zeros(self.max_iter)

            for idx_iter in range(self.max_iter):
                err = X_ @ theta - y

                # Calculate the J term, which is the current MSE
                J[idx_iter] = (0.5/m) * err.T @ err

                # The gradient
                grad = (1/m) * np.conjugate(X_).T @ err

                # Here is the actual update
                theta = theta - lr * grad
            res[lr] = dict(result=theta, cost=J)

        self.res_ = res
        idx_lower_cost = np.argmin([v['cost'][-1] for k, v in res.items()])
        self.best_lr_ = self.alpha[idx_lower_cost]

        return self

    def predict(self, X):
        """Multiply the feature matrix with the optimum vector."""
        X_ = LMS.add_intercept_term(X)
        if self.scale:
            X_[:, 1:] = (X_[:, 1:] - self.mu_[1:]) / self.std_[1:]
        best_theta = self.res_[self.best_lr_]['result']
        return X_ @ best_theta
