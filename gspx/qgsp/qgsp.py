"""Utils related to Quaternion-GSP."""

from ast import Assert
import numpy as np
from pyquaternion import Quaternion
import sys

from gspx.utils.quaternion_matrix import complex_adjoint, \
    implode_quaternions, conjugate, explode_quaternions, \
    from_complex_to_exploded, from_exploded_to_complex
from gspx.utils.display import visualize_quat_mtx
from gspx.utils.quaternion_matrix import quaternion_array


class QMatrix:
    """Deal with quaternion matrices in the context of graph matrices.

    Parameters
    ----------
    tup : sequence of 4 arrays with same shape
        The arrays must have the same shape and the sequence
        must contain no more than 4 arrays. Each array
        contains one of the 4 quaternion dimensions.

    Attribute
    ---------
    matrix : quaternion-valued 2D array of shape (N, M)

    Properties
    ----------
    shape : tuple
    complex_adjoint : complex-valued array of shape (2*N, 2*M)

    """

    matrix = None

    def __init__(self, tup=None):
        """Construct."""
        if tup is None:
            return
        assert isinstance(tup, (list, tuple)) and len(tup) <= 4
        shape = tup[0].shape
        arrays = list(tup)
        while len(arrays) < 4:
            arrays.append(np.zeros(shape))
        if tup is not None:
            self.matrix = quaternion_array(np.vstack(arrays).transpose())

    @staticmethod
    def from_matrix(qmatrix):
        """Instantiate an object with the given quaternion matrix."""
        new = QMatrix()
        new.matrix = qmatrix
        return new

    def to_sparse(self):
        r"""Write the object as a list of non-zero entries and its indices.

        Example
        -------
        >>> import numpy as np
        >>> rnd = np.random.default_rng(seed=2)
        >>> mat = QMatrix([np.ones((3, 3)) for _ in range(4)])
        >>> mat.matrix[:, 1:] = Quaternion(0, 0, 0, 0)
        >>> mat
        Quaternion-valued array of shape (3, 3):
        [[Quaternion(1.0, 1.0, 1.0, 1.0) Quaternion(0.0, 0.0, 0.0, 0.0)
        Quaternion(0.0, 0.0, 0.0, 0.0)]
        [Quaternion(1.0, 1.0, 1.0, 1.0) Quaternion(0.0, 0.0, 0.0, 0.0)
        Quaternion(0.0, 0.0, 0.0, 0.0)]
        [Quaternion(1.0, 1.0, 1.0, 1.0) Quaternion(0.0, 0.0, 0.0, 0.0)
        Quaternion(0.0, 0.0, 0.0, 0.0)]]

        >>> entries, idx_nz, shape = mat.to_sparse()
        >>> print(
        ...     f"Non-zero Entries: {entries}\nIndices: "
        ...     f"{idx_nz}\nShape: {shape}")
        Non-zero Entries: [
            Quaternion(1.0, 1.0, 1.0, 1.0)
            Quaternion(1.0, 1.0, 1.0, 1.0)
            Quaternion(1.0, 1.0, 1.0, 1.0)
        ]
        Indices: (array([0, 1, 2]), array([0, 0, 0]))
        Shape: (3, 3)

        """
        idx_nz = np.where(self.matrix != Quaternion(0, 0, 0, 0))
        entries = self.matrix[idx_nz]
        return entries, idx_nz, self.shape

    @staticmethod
    def from_sparse(entries, idx_nz, shape):
        """Retrieve the object from its sparse parameters."""
        new = np.zeros(shape, dtype=entries.dtype) + Quaternion(0, 0, 0, 0)
        new[idx_nz] = entries
        return QMatrix.from_matrix(new)

    @staticmethod
    def vander(signal, deg, increasing=True):
        """Create a Vandermonde quaternion matrix."""
        assert len(signal.matrix) == len(signal.matrix.ravel()), (
            "Provide a QMatrix column vector or a QuaternionSignal."
        )
        arr = signal.matrix.ravel()
        mat = np.vander(arr, N=deg, increasing=increasing)
        mat[:, 0] = [Quaternion(1, 0, 0, 0) for _ in arr]
        return QMatrix.from_matrix(mat)

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

    def mean(self, axis=0):
        """Mean accross an axis."""
        return self.matrix.mean(axis=axis)

    def copy(self):
        """Copy of the object."""
        return QMatrix.from_matrix(self.matrix)

    @staticmethod
    def visualize(obj, dpi=150):
        """Visualize the quaternion matrix."""
        visualize_quat_mtx(obj.matrix, dpi=dpi)

    def __repr__(self):
        """Represent as string."""
        msg = (
            "\x1B[3mQuaternion-valued array of shape "
            f"{self.shape}:\x1B[0m\n{self.matrix}"
        )
        return msg

    def __add__(self, other):
        """Add."""
        if isinstance(other, QMatrix):
            return QMatrix.from_matrix(self.matrix + other.matrix)

        return QMatrix.from_matrix(self.matrix + other)

    def __sub__(self, other):
        """Subtract."""
        return self + other * (-1)

    def __mul__(self, other):
        """Multiply."""
        if isinstance(other, QMatrix):
            # We use multiplication between the complex representations
            # of each quaternion matrix
            prod = from_exploded_to_complex(
                explode_quaternions(self.matrix)
            ) @ from_exploded_to_complex(
                explode_quaternions(other.matrix)
            )
            prodq = implode_quaternions(from_complex_to_exploded(prod))
            return QMatrix.from_matrix(prodq)

        return QMatrix.from_matrix(self.matrix * other)

    def __div__(self, other):
        """Divide."""
        return QMatrix.from_matrix(self.matrix / other)

    def __eq__(self, other):
        """Assert equality."""
        return (self.matrix == other.matrix).all()

    def __getitem__(self, key):
        """Get item."""
        return QMatrix.from_matrix(self.matrix[key])

    def hadamard(self, other):
        """Perform element-wise product."""
        assert self.matrix.shape == other.matrix.shape
        return QMatrix.from_matrix(self.matrix * other.matrix)

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
        eigq = QMatrix.from_matrix(eigq[:, np.newaxis])

        return eigq, QMatrix.from_matrix(Vq)

    def inv(self, rtol=1e-05, atol=1e-08):
        """Compute the matrix inverse, if exists."""
        ca = self.complex_adjoint
        assert np.linalg.cond(ca) < 1 / sys.float_info.epsilon, (
            "The given matrix is not invertible."
        )
        nrows, _ = ca.shape
        n = int(nrows / 2)
        ca_inv = np.linalg.inv(ca)
        inv00 = ca_inv[:n, :n]
        inv01 = ca_inv[:n, n:]
        inv10 = ca_inv[n:, :n]
        inv11 = ca_inv[n:, n:]

        # Checking if the inverse of the complex adjoint also has the form
        # of a complex adjoint
        assert np.allclose(inv00, inv11.conjugate(), rtol=rtol, atol=atol)
        assert np.allclose(inv01, - inv10.conjugate(), rtol=rtol, atol=atol)

        qmatrix = implode_quaternions(np.dstack((
            np.real(inv00),
            np.imag(inv00),
            np.real(inv01),
            np.imag(inv01),
        )))
        return QMatrix.from_matrix(qmatrix)


class QGFT:
    """Quaternion-valued Graph Fourier Transform."""

    def __init__(self, verbose=True, sort=True):
        """Construct."""
        self.verbose = verbose
        self.eigq = None
        self.eigc = None
        self.Vq = None
        self.Vq_inv = None
        self.sort = sort
        self.idx_freq = None
        self.tv_ = None

    def inform(self, msg):
        """Inform the user."""
        if self.verbose:
            print(msg)

    def fit(self, shift_operator):
        """Fit the object.

        Parameters
        ----------
        shift_operator : np.ndarray, shape=(N,)

        """
        assert isinstance(shift_operator, QMatrix), (
            "The shift operator is expected to be a QMatrix."
        )

        self.inform("Running eigendecomposition of the shift operator.")
        self.eigq, self.Vq = shift_operator.eigendecompose()
        try:
            self.Vq_inv = self.Vq.inv()
        except AssertionError as e:
            self.inform(
                f"The eigenvector matrix could not be inverted: {e}.")

        # Storing a complex-valued copy of the quaternionic eigenvalues
        self.eigc = (
            explode_quaternions(self.eigq.matrix)[:, :, 0] +
            1j * explode_quaternions(self.eigq.matrix)[:, :, 1])

        # Frequency ordering
        if self.sort:
            self.inform("Sorting the frequencies based on Total Variation.")
            self.idx_freq, self.tv_ = self.sort_frequencies(shift_operator)

        return self

    def sort_frequencies(self, shift_operator):
        """Find the eigenvalues order that sort the frequencies."""
        assert self.Vq is not None, ("One must run `fit` first.")
        Vq_shifted = shift_operator * self.Vq
        diff = Vq_shifted - self.Vq
        diff_norm_squared = (diff.transpose().conjugate() * diff).diag()
        tv = np.sqrt(np.abs(diff_norm_squared).astype(float))
        return np.argsort(tv), tv

    def transform(self, signal):
        """Apply the direct QGFT.

        Parameters
        ----------
        signal : np.ndarray, shape=(N,)

        """
        assert self.Vq_inv is not None, (
            "The eigenvector matrix was not inverted."
        )
        assert len(signal.matrix) == len(signal.matrix.ravel()), (
            "Provide a QMatrix column vector or a QuaternionSignal."
        )
        ss = self.Vq_inv * signal
        return ss

    def inverse_transform(self, signal):
        """Apply the inverse QGFT.

        Parameters
        ----------
        signal : np.ndarray, shape=(N,)

        """
        assert len(signal.matrix) == len(signal.matrix.ravel()), (
            "Provide a QMatrix column vector or a QuaternionSignal."
        )
        ss = self.Vq * signal
        return ss
