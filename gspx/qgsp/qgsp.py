"""Utils related to Quaternion-GSP."""

import sys
import numpy as np
from pyquaternion import Quaternion

from gspx.utils.quaternion_matrix import complex_adjoint, \
    implode_quaternions, conjugate, explode_quaternions, \
    from_complex_to_exploded, from_exploded_to_complex
from gspx.utils.display import visualize_quat_mtx


class QMatrix:
    """Deal with quaternion matrices in the context of graph matrices.

    Parameters
    ----------
    seq : sequence of 4 arrays with same shape
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

    def __init__(self, seq=None):
        """Construct."""
        if seq is None:
            return
        assert isinstance(seq, (list, tuple, np.ndarray)) and len(seq) <= 4
        shape = np.array(seq[0]).shape
        arrays = list(seq)
        while len(arrays) < 4:
            arrays.append(np.zeros(shape))
        self.matrix = implode_quaternions(np.dstack(arrays))

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
        >>> entries, idx_nz, shape = mat.to_sparse()
        >>> entries
        array([Quaternion(1.0, 1.0, 1.0, 1.0), Quaternion(1.0, 1.0, 1.0, 1.0),
               Quaternion(1.0, 1.0, 1.0, 1.0)], dtype=object)

        >>> print(
        ...     f"Indices: {idx_nz}\nShape: {shape}")
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
            "Quaternion-valued array of shape "
            f"{self.shape}:\n{self.matrix}"
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

    @staticmethod
    def unstack_complex_eig_matrix(V: np.ndarray, eig: np.ndarray):
        """Unstack the complex adjoint eigenvector matrix."""
        eig = np.round(eig, decimals=10)
        mask_imag_pos = np.imag(eig) > 0
        mask_imag_null = np.imag(eig) == 0
        track = []
        for i, e in enumerate(eig[mask_imag_null]):
            if e in track:
                mask_imag_null[i] = False
            else:
                track.append(e)

        # Taking only half of the eigenvalues with null imaginary part
        how_many_null = int((len(eig) - len(np.where(mask_imag_pos)[0])) / 2)
        idx = np.where(mask_imag_null)[0]
        mask_imag_null[idx[how_many_null:]] = False

        mask = np.logical_or(mask_imag_pos, mask_imag_null)
        V1 = V[:int(len(V) / 2), mask]
        V2 = V[int(len(V) / 2):, mask]

        Vq = implode_quaternions(np.dstack((
            np.real(V1),
            np.imag(V1),
            - np.real(V2),
            np.imag(V2)
        )))
        return Vq, mask

    def eigendecompose(self, hermitian_gso=False):
        """Eigendecompose the input matrix.

        Parameters
        ----------
        hermitian_gso : bool, default=True

        Return
        ------
        eigq : np.ndarray with Quaternion entries.
        Vq : QMatrix
            Matrix with eigenvectors.

        """
        if not hermitian_gso:
            eig, V = np.linalg.eig(self.complex_adjoint)
        else:
            eig, V = np.linalg.eigh(self.complex_adjoint)

        Vq, mask = QMatrix.unstack_complex_eig_matrix(V=V, eig=eig)

        eigq = (
            np.real(eig[mask]) * Quaternion(1, 0, 0, 0) +
            np.imag(eig[mask]) * Quaternion(0, 1, 0, 0)
        )
        eigq = QMatrix.from_matrix(eigq[:, np.newaxis])

        return eigq, QMatrix.from_matrix(Vq)

    def is_hermitian(self, tol: float = 1e-8, slice_frac: float = 0.5):
        """Check if the quaternion matrix is hermitian.

        Parameters
        ----------
        tol : float, default=1e-8
            Tolerance.
        slice_frac : float between 0 and 1, default=0.5
            Fraction of the number of rows and columns that will be checked.
            The idea is that we try to make this verification faster, at the
            cost of looking only at a sample of the matrix.

        """
        if slice_frac == 1:
            obj = self
        else:
            idx = np.random.permutation(len(self.matrix))
            idx = idx[:int(slice_frac * len(idx))]
            obj = QMatrix.from_matrix(self.matrix[idx, :][:, idx])
        return np.all((obj - obj.conjugate().transpose()).abs() < tol)

    def inv(
            self, rtol: float = 1e-05, atol: float = 1e-08,
            assert_cond: bool = True):
        """Compute the matrix inverse, if exists."""
        ca = self.complex_adjoint

        if not self.is_hermitian() and assert_cond:
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

    def __init__(
            self, verbose: bool = True, sort: bool = True,
            hermitian_gso: bool = True, norm: int = 1):
        """Construct."""
        norm_values = [1, 2]
        assert norm in norm_values, (
            f"The norm currently must be one of the values: {norm_values}."
        )
        self.verbose = verbose
        self.hermitian_gso = hermitian_gso
        self.norm = norm
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
        self.eigq, self.Vq = shift_operator.eigendecompose(
            hermitian_gso=self.hermitian_gso)

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

    def sort_frequencies(self, shift_operator: QMatrix):
        """Find the eigenvalues order that sort the frequencies."""
        assert self.Vq is not None, ("One must run `fit` first.")

        Vq_shifted = shift_operator * self.Vq
        diff = Vq_shifted - self.Vq
        if self.norm == 1:
            tv = [
                np.sum([q.norm for q in diff.matrix[:, i]])
                for i in range(len(diff.matrix))
            ]
        elif self.norm == 2:
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
