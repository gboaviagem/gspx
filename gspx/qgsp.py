"""Utils related to Quaternion-GSP."""

import numpy as np
from pyquaternion import Quaternion

from gspx.signals import QuaternionSignal
from gspx.utils.quaternion_matrix import complex_adjoint, \
    implode_quaternions, conjugate
from gspx.utils.display import visualize_quat_mtx


class QMatrix:
    """Deal with quaternion matrices in the context of graph matrices.

    Parameters
    ----------
    tup : sequence of 2D arrays
        The arrays must have the same shape and the sequence
        must contain no more than 4 arrays.

    Attribute
    ---------
    matrix : quaternion-valued 2D array of shape (N, M)

    Properties
    ----------
    shape : tuple
    complex_adjoint : complex-valued array of shape (2*N, 2*M)

    """

    def __init__(self, tup):
        """Construct."""
        assert isinstance(tup, (list, tuple)) and len(tup) <= 4
        shape = tup[0].shape
        arrays = list(tup)
        while len(arrays) < 4:
            arrays.append(np.zeros(shape))

        QS = QuaternionSignal.from_rectangular(np.hstack(
            tuple([arr.ravel()[:, np.newaxis] for arr in arrays])
        ))
        self.matrix = QS.samples.reshape(shape)

    @staticmethod
    def from_matrix(qmatrix):
        """Instantiate an object with the given quaternion matrix."""
        Q = QMatrix([np.ones(qmatrix.shape)] * 4)
        Q.matrix = qmatrix
        return Q

    @property
    def shape(self):
        """Get the matrix shape."""
        return self.matrix.shape

    @property
    def complex_adjoint(self):
        """Build the matrix complex adjoint."""
        return complex_adjoint(self.matrix)

    def abs(self):
        """Absolute value (element-wise)."""
        return np.abs(self.matrix).astype(float)

    def conjugate(self):
        """Conjugate of the matrix."""
        return QMatrix.from_matrix(conjugate(self.matrix))

    def transpose(self):
        """Transpose."""
        return QMatrix.from_matrix(self.matrix.transpose())

    def diag(self):
        """Get the diagonal od the matrix."""
        return np.diag(self.matrix)

    def visualize(self, dpi=150):
        """Visualize the quaternion matrix."""
        visualize_quat_mtx(self.matrix, dpi=dpi)

    def __repr__(self):
        """Represent as string."""
        msg = (
            "\x1B[3m"
            "Quaternion-valued array of shape "
            "{}:\x1B[0m\n{}".format(self.shape, self.matrix)
        )
        return msg

    def __add__(self, other):
        """Add."""
        if isinstance(other, QMatrix):
            return QMatrix.from_matrix(self.matrix + other.matrix)
        else:
            return QMatrix.from_matrix(self.matrix + other)

    def __sub__(self, other):
        """Subtract."""
        return self + other * (-1)

    def __mul__(self, other):
        """Multiply."""
        if isinstance(other, QMatrix):
            return QMatrix.from_matrix(self.matrix @ other.matrix)
        else:
            return QMatrix.from_matrix(self.matrix * other)

    def __eq__(self, other):
        """Assert equality."""
        return (self.matrix == other.matrix).all()

    def __getitem__(self, key):
        """Get item."""
        return QMatrix.from_matrix(self.matrix[key])

    def eigendecompose(self):
        """Eigendecompose the input matrix.

        Return
        ------
        eigq : np.ndarray with Quaternion entries.
        Vq : QMatrix
            Matrix with eigenvectors.

        """
        eig, V = np.linalg.eig(self.complex_adjoint)

        mask_imag_pos = np.imag(eig) > 0
        mask_imag_null = np.imag(eig) == 0

        # Taking only half of the eigenvalues with null imaginary part
        idx = np.where(mask_imag_null)[0]
        mask_imag_null[idx[:int(len(idx)/2)]] = False

        mask = np.logical_or(mask_imag_pos, mask_imag_null)
        V1 = V[:int(len(V) / 2), mask]
        V2 = V[int(len(V) / 2):, mask]

        Vq = implode_quaternions(np.dstack((
            np.real(V1),
            np.imag(V1),
            - np.real(V2),
            np.imag(V2)
        )))

        eigq = (
            np.real(eig[mask]) * Quaternion(1, 0, 0, 0) +
            np.imag(eig[mask]) * Quaternion(0, 1, 0, 0)
        )

        return eigq, QMatrix.from_matrix(Vq)


class qGFT:
    """Quaternion-valued Graph Fourier Transform."""

    def __init__(self):
        """Construct."""
        pass

    def fit(self, shift_operator):
        """Fit the object.

        Parameters
        ----------
        shift_operator : np.ndarray, shape=(N,)

        """
        assert isinstance(shift_operator, QMatrix), (
            "The shift operator is expected to be a QMatrix."
        )

        self.eigq, self.Vq = shift_operator.eigendecompose()

        new = QuaternionSignal()
        new.samples = self.eigq
        # Storing a complex-valued copy of the quaternionic eigenvalues
        self.eigc = new.to_array()[:, 0] + 1j * new.to_array()[:, 1]

        return self

    def transform(self, signal):
        """Apply the direct qGFT.

        Parameters
        ----------
        shift_operator : np.ndarray, shape=(N,)

        """
        assert isinstance(signal, np.ndarray), (
            "The signal is expected to be a Numpy 1D array."
        )
        return self.Vq.matrix @ signal.ravel()[:, np.newaxis]

    def inverse_transform(self, signal):
        """Apply the inverse qGFT.

        Parameters
        ----------
        shift_operator : np.ndarray, shape=(N,)

        """
        assert isinstance(signal, np.ndarray), (
            "The signal is expected to be a Numpy 1D array."
        )
        return self.Vq.matrix @ signal.ravel()[:, np.newaxis]
