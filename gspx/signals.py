"""Graph signal class."""

import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from gspx.qgsp import QMatrix
from gspx.base import Signal
from gspx.utils.quaternion_matrix import quaternion_array


class QuaternionSignal(QMatrix, Signal):
    r"""Representation of a 1D quaternion-valued signal.

    Parameters
    ----------
    seq : sequence of 4 sequences of real numbers with same length
        Each sequence corresponds to a quaternion dimension: first the
        sequence of real parts, then the sequence of "i" coefficients,
        then the "j" coefficients, and finally the "k" coefficients.

    Example
    -------
    >>> # In order to create the array [1 + 0i + 3j + 4k, 2 + 5i + 3j + 7k]
    >>> # one may call
    >>> QuaternionSignal([
    ...     (1, 2),
    ...     (0, 5),
    ...     (3, 3),
    ...     (4, 7)
    ... ])
    Quaternion-valued array of shape (2, 1):
    [[Quaternion(1.0, 0.0, 3.0, 4.0)]
     [Quaternion(2.0, 5.0, 3.0, 7.0)]]

    >>> # or even through
    >>> QuaternionSignal.from_rectangular([[1, 0, 3, 4], [2, 5, 3, 7]])
    Quaternion-valued array of shape (2, 1):
    [[Quaternion(1.0, 0.0, 3.0, 4.0)]
     [Quaternion(2.0, 5.0, 3.0, 7.0)]]

    """

    def __init__(self, seq=None):
        """Construct."""
        if seq is not None:
            assert len(set([len(s) for s in seq])) == 1
            seq = [np.array(s)[:, np.newaxis] for s in seq]
        QMatrix.__init__(self, seq=seq)

    def conjugate(self, inplace=False):
        r"""Return the conjugate of samples.

        Parameters
        ----------
        inplace : bool, default=False
            Whether the conjugation will affect the object samples, or
            will create a new instance.

        Example
        -------
        >>> s = QuaternionSignal.from_rectangular([
        ...     [1, 2, 3, 4],
        ...     [2, -3, -4, 1]
        ... ])
        >>> s.conjugate(inplace=False)
        Quaternion-valued array of shape (2, 1):
        [[Quaternion(1.0, -2.0, -3.0, -4.0)]
         [Quaternion(2.0, 3.0, 4.0, -1.0)]]

        """
        new_samples = np.array([q.conjugate for q in self.matrix.ravel()])
        if inplace:
            self.matrix = new_samples[:, np.newaxis]
        else:
            return QuaternionSignal.from_samples(new_samples)

    def involution(self, axis="i", inplace=False):
        r"""Compute the involution of each sample by a given axis.

        Parameters
        ----------
        axis : array-like of shape=(4,) or {'i', 'j', 'k'}, default='i'
            The axis used in the involution. The unit quaternion `mu`
            in the given axis is used in the computation
            `- mu * self.matrix * mu`. The default is [0, 1, 0, 0].
        inplace : bool, default=False
            Whether the involution will affect the object samples, or
            will create a new instance.

        Example
        -------
        >>> from gspx.signals import QuaternionSignal
        >>> arr = [[1, 2, 3, 4], [2, -3, -4, 1]]
        >>> s = QuaternionSignal.from_rectangular(arr)
        >>> s.involution("i", inplace=False)
        Quaternion-valued array of shape (2, 1):
        [[Quaternion(1.0, 2.0, -3.0, -4.0)]
         [Quaternion(2.0, -3.0, 4.0, -1.0)]]

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

        shape = self.matrix.shape
        mu = Quaternion(axis).unit
        new_samples = np.array([- mu * q * mu for q in self.matrix.ravel()])
        if inplace:
            self.matrix = new_samples.reshape(shape)
        else:
            return QuaternionSignal.from_samples(new_samples)

    @staticmethod
    def from_samples(new_samples):
        """Create an instance out of a quaternionic array."""
        new = QuaternionSignal()
        new.matrix = new_samples[:, np.newaxis]
        return new

    @staticmethod
    def from_rectangular(arr, def_real_part=1.0, dtype=None):
        """Create a QuaternionSignal from the rectangular form.

        Parameters
        ----------
        arr : array-like, shape=(N, 4)

        """
        samples = quaternion_array(
            arr, def_real_part=def_real_part, dtype=dtype)
        return QuaternionSignal.from_samples(samples)

    @staticmethod
    def from_equal_dimensions(arr):
        """Create a QuaternionSignal in which all dimensions are equal."""
        return QuaternionSignal.from_rectangular(
            np.hstack([arr.ravel()[:, np.newaxis]] * 4)
        )

    def to_array(self, max_value=None, **kwargs):
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
            [a[0], a[1], a[2], a[3]] for a in self.matrix.ravel()
        ])
        if max_value == 'self':
            max_value = out.max()
        if max_value is not None:
            out = out / max_value
        return out

    def __len__(self):
        """Compute length."""
        return len(self.matrix)

    @property
    def norm1(self) -> float:
        """Return the L1-norm of this signal."""
        return np.sum([q.norm for q in self.matrix.ravel()])

    @property
    def norm2(self) -> float:
        """Return the L2-norm of this signal."""
        return np.sqrt(
            (self.conjugate().transpose() * self).matrix.ravel()[0].norm)

    def to_rgb(self, **kwargs):
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

    @staticmethod
    def show(obj, dpi=200, ordering=None, **kwargs):
        """Visualize the signal quaternion dimensions."""
        arr1D = obj.matrix.ravel()
        if ordering is not None:
            assert isinstance(ordering, (np.ndarray, list))
            arr1D = arr1D[ordering]
        arr = QuaternionSignal.from_samples(arr1D).to_array()
        fig, axs = plt.subplots(2, 2, dpi=dpi)
        titles = ["Real part", "i-component", "j-component", "k-component"]
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        params = dict(
            marker='.', markersize=5,
            alpha=0.8, linestyle='dotted',
            linewidth=1.0
        )
        params.update(kwargs)
        for i in range(4):
            axs[int(i > 1), i % 2].plot(arr[:, i], color=colors[i], **params)
            axs[int(i > 1), i % 2].set_title(titles[i])
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.show()
        return fig
