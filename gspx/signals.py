"""Graph signal class."""

import numpy as np
from pyquaternion import Quaternion
from gspx.base import Signal


class QuaternionSignal(Signal):
    """Representation of a 1D quaternion-valued signal.

    Parameters
    ----------
    values : list of dict
        List containing data to instantiate one pyquaternion.Quaternion
        at a time. The data consist of many dicts with keys being
        keyword arguments of the pyquaternion.Quaternion class.
        Options of keys for each method of instantiation:

        - "scalar" and "vector"
        - "real" and "imaginary"
        - "axis" and "radians", or "degree", or "angle"
        - "array"
        - "matrix"

    """

    def __init__(self, values):
        """Construct."""
        if isinstance(values, dict):
            values = [values]

        self.samples = np.array([
            Quaternion(**kwargs) for kwargs in values
        ])

    @staticmethod
    def from_rectangular(arr):
        """Create a QuaternionSignal from the rectangular form.

        Parameters
        ----------
        arr : array-like, shape=(N, 4)

        """
        dims = np.array(arr).shape
        assert len(dims) == 2 and dims[1] == 4, (
            "Please provide a (N, 4) array-like object."
        )
        return QuaternionSignal([dict(array=row) for row in arr])

    def to_rgba(self, max_value=None):
        """Create the RGB representation.

        Parameters
        ----------
        max_value : float, default=None

        """
        out = np.array([
            [a[0], a[1], a[2], a[3]] for a in self.samples
        ])
        if max_value is None:
            max_value = out.max()
        return out / max_value
