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

    samples = None

    def __init__(self, values=None):
        """Construct."""
        if isinstance(values, dict):
            values = [values]

        if values is not None:
            self.samples = np.array([
                Quaternion(**kwargs) for kwargs in values
            ])

    def conjugate(self, inplace=False):
        """Return the conjugate of samples.

        Parameters
        ----------
        inplace : bool, default=False
            Whether the conjugation will affect the object samples, or
            will create a new instance.

        Example
        -------
        >>> from gspx.signals import QuaternionSignal
        >>> arr = [[1, 2, 3, 4], [2, -3, -4, 1]]
        >>> s = QuaternionSignal.from_rectangular(arr)
        >>> q = s.conjugate(inplace=False)
        >>> q.samples.tolist()
        [Quaternion(1.0, -2.0, -3.0, -4.0), Quaternion(2.0, 3.0, 4.0, -1.0)]

        """
        new_samples = np.array([q.conjugate for q in self.samples])
        if inplace:
            self.samples = new_samples
        else:
            new = QuaternionSignal()
            new.samples = new_samples
            return new

    def involution(self, axis=[0, 1, 0, 0], inplace=False):
        """Compute the involution of each sample by a given axis.

        Parameters
        ----------
        axis : array-like of shape=(4,) or str {'i', 'j', 'k'}
            The axis used in the involution. The unit quaternion `mu`
            in the given axis is used in the computation
            `- mu * self.samples * mu`. The default is [0, 1, 0, 0].
        inplace : bool, default=False
            Whether the involution will affect the object samples, or
            will create a new instance.

        Example
        -------
        >>> from gspx.signals import QuaternionSignal
        >>> arr = [[1, 2, 3, 4], [2, -3, -4, 1]]
        >>> s = QuaternionSignal.from_rectangular(arr)
        >>> q = s.involution("i", inplace=False)
        >>> q.samples.tolist()
        [Quaternion(1.0, 2.0, -3.0, -4.0), Quaternion(2.0, -3.0, 4.0, -1.0)]

        """
        if axis == "i":
            axis = [0, 1, 0, 0]
        elif axis == "j":
            axis = [0, 0, 1, 0]
        elif axis == "k":
            axis = [0, 0, 0, 1]
        else:
            assert len(axis) == 4, (
                "The `axis` must be either an array-like of length 4 "
                "of one of the characters: [i, j, k], indicating the "
                "ordinary basis for pure quaternions."
            )

        mu = Quaternion(axis).unit
        new_samples = np.array([- mu * q * mu for q in self.samples])
        if inplace:
            self.samples = new_samples
        else:
            new = QuaternionSignal()
            new.samples = new_samples
            return new

    @property
    def shape(self):
        """Return the shape of the array of samples."""
        return self.samples.shape

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

    def to_array(self, max_value=None):
        """Create a pure-numpy array representation.

        The signal with length N turns into a (N, 4) float-valued numpy array.

        The r, g and b channels are mapped to the vector components
        of the quaternion, whereas the alpha channel is the real part.

        Parameters
        ----------
        max_value : str {'self'} or float, default=None
            All the channel values are divided by `max_value`. If None,
            division is not performed. If 'self', the maximum value of the
            numpy representation is used.

        """
        out = np.array([
            [a[0], a[1], a[2], a[3]] for a in self.samples
        ])
        if max_value == 'self':
            max_value = out.max()
        if max_value is not None:
            out = out / max_value
        return out

    def to_rgb(self):
        """RGB (normalized) representation of the signal. Real part ignored."""
        arr = self.to_array(max_value='self')
        return arr[:, 1:]

    def to_rgba(self):
        """RGBA (normalized) representation of the signal."""
        def normalize(array):
            array -= np.min(array)
            return array / np.max(array)

        arr = self.to_array(max_value=None)
        # arr = arr - np.min(arr)
        # arr = arr / np.max(arr)
        rgba = np.zeros(arr.shape, dtype=arr.dtype)
        rgba[:, 0:3] = normalize(arr[:, 1:4].copy())
        rgba[:, 3] = normalize(arr[:, 0].copy())
        return rgba
