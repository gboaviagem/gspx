## Quaternion matrices
Using the class `QMatrix`, one can manipulate quaternion matrices
with methods suitable for graph operations.

One can instantiate the object either from four real-valued matrices,
each representing one quaternionic dimension (`1`, `i`, `j` and `k`),
```py
>>> import numpy as np
>>> from gspx.qgsp import QMatrix
>>> M1 = np.random.default_rng(seed=2).integers(2, size=(3, 3))
>>> Mi = np.random.default_rng(seed=3).integers(2, size=(3, 3))
>>> Mj = np.random.default_rng(seed=4).integers(2, size=(3, 3))
>>> Mk = np.random.default_rng(seed=5).integers(2, size=(3, 3))
>>> M = QMatrix([M1, Mi, Mj, Mk])
>>> M
Quaternion-valued array of shape (3, 3):
[[Quaternion(1.0, 1.0, 1.0, 1.0) Quaternion(0.0, 0.0, 1.0, 1.0)
  Quaternion(0.0, 0.0, 1.0, 0.0)]
 [Quaternion(0.0, 0.0, 1.0, 1.0) Quaternion(0.0, 0.0, 1.0, 0.0)
  Quaternion(1.0, 1.0, 1.0, 1.0)]
 [Quaternion(0.0, 1.0, 1.0, 1.0) Quaternion(0.0, 1.0, 0.0, 0.0)
  Quaternion(0.0, 0.0, 0.0, 1.0)]]

>>> # Experiment a visual inspection of the matrix:
>>> M.visualize()
```
or one can create it from a sparse representation, in which each
non-zero entry is a `pyquaternion.Quaternion` instance:
```py
>>> import numpy as np
>>> from pyquaternion import Quaternion
>>> from gspx.qgsp import QMatrix
>>> entries = np.array([
...     Quaternion(0, 2, 2, 0),
...     Quaternion(2, 2, 0, 0)
... ])
>>> idx_nz = (np.array([1, 2]), np.array([1, 1]))
>>> shape = (3, 2)
>>> M = QMatrix.from_sparse(entries, idx_nz, shape)
>>> M
Quaternion-valued array of shape (3, 2):
[[Quaternion(0.0, 0.0, 0.0, 0.0) Quaternion(0.0, 0.0, 0.0, 0.0)]
 [Quaternion(0.0, 0.0, 0.0, 0.0) Quaternion(0.0, 2.0, 2.0, 0.0)]
 [Quaternion(0.0, 0.0, 0.0, 0.0) Quaternion(2.0, 2.0, 0.0, 0.0)]]
```

The arithmetic special methods are also implemented (in which the
multiplication is performed in the matrix sense, not element-wise). See this example of multiplication:
```py
>>> M1
Quaternion-valued array of shape (3, 2):
[[Quaternion(0.0, 0.0, 0.0, 0.0) Quaternion(0.0, 0.0, 0.0, 0.0)]
 [Quaternion(0.0, 0.0, 0.0, 0.0) Quaternion(0.0, 2.0, 2.0, 0.0)]
 [Quaternion(0.0, 0.0, 0.0, 0.0) Quaternion(2.0, 2.0, 0.0, 0.0)]]
>>> M2
Quaternion-valued array of shape (2, 1):
[[Quaternion(0.0, 0.0, 0.0, 0.0)]
 [Quaternion(1.0, 1.0, 1.0, 1.0)]]
>>> M1 * M2
Quaternion-valued array of shape (3, 1):
[[Quaternion(0.0, 0.0, 0.0, 0.0)]
 [Quaternion(-4.0, 4.0, 0.0, 0.0)]
 [Quaternion(0.0, 4.0, 0.0, 4.0)]]
```

The two most important methods, in the context of graph operations,
are the creation of the `complex adjoint` matrix and the `eigendecomposition`.
```py
>>> from gspx.utils.graph import make_sensor
>>> from gspx.qgsp import QMatrix
>>> nvertices = 6
>>> # Let us create four adjacency matrices of sensor graphs
>>> A1, coords = make_sensor(N=nvertices, seed=2)
>>> Ai, _ = make_sensor(N=nvertices, seed=3)
>>> Aj, _ = make_sensor(N=nvertices, seed=4)
>>> Ak, _ = make_sensor(N=nvertices, seed=5)
>>> A = QMatrix([A1, Ai, Aj, Ak])

# The eigendecomposition returns a quaternionic array with the
# standard right eigenvalues, and a QMatrix of eigenvectors
>>> eigq, Vq = A.eigendecompose()

# The complex adjoint can be extracted
>>> ca = A.complex_adjoint
>>> ca.shape
(12, 12)
```