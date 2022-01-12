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
    def from_rectangular(arr, def_real_part=1.0, dtype=None):
        """Create a QuaternionSignal from the rectangular form.

        Parameters
        ----------
        arr : array-like, shape=(N, 4)

        """
        dims = np.array(arr).shape
        assert len(dims) == 2 and dims[1] in [3, 4], (
            "Please provide a (N, 3) or (N, 4) array-like object."
        )
        if dims[1] == 3:
            # If the quaternion real part is not provided, we set it to
            # 1, for the sake of convenience with color image signals.
            arr = np.hstack((
                np.full((dims[0], 1), fill_value=def_real_part, dtype=dtype),
                arr
            ))
        return QuaternionSignal([dict(array=row) for row in arr])

    def to_rgba(self, max_value=None):
        """Create the rgba representation.

        The r, g and b channels are mapped to the vector components
        of the quaternion, whereas the alpha channel is the real part.

        Parameters
        ----------
        max_value : float, default=None
            All the channel values are divided by `max_value`. If None,
            then it is the maximum value in the (N, 4) resulting array.

        """
        out = np.array([
            [a[1], a[2], a[3], a[0]] for a in self.samples
        ])
        if max_value is None:
            max_value = out.max()
        return out / max_value
