"""Simple example on first uses of QGSP."""

# If gspx is not installed, we add it to the path
import os, sys
gdir = os.getcwd()
sys.path.insert(0, gdir)

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion

from gspx.utils.display import plot_graph
from gspx.qgsp import QMatrix
from gspx.utils.quaternion_matrix import \
    explode_quaternions, implode_quaternions
from gspx.qgsp import QGFT

# Let us say we have the coordinates of some graph vertices
coords = np.array([
    [0, 1],
    [-1, 0],
    [0, -1],
    [1, 0],
    [0, 0],
])

# Indices of non-zero adjacency matrix entries
idx = np.array([
    [0, 4],
    [1, 4],
    [2, 4],
    [3, 4],
    [3, 0]
])

# Filling the adjacency matrix
Aq = QMatrix.from_matrix(implode_quaternions(np.zeros((5, 5, 4))))
Ne = 5
rnd = np.random.RandomState(seed=42)
entries = rnd.randint(10, size=(Ne, 4))

for n, i in enumerate(idx):
    Aq.matrix[i[0], i[1]] = Quaternion(entries[n])

# Symmetric graph
Aq = Aq + Aq.conjugate().transpose()

eigq, Vq = Aq.eigendecompose(hermitian_gso=True)

print("Adjacency matrix rows:")
for row in Aq.matrix:
    print([str(q) for q in row])

# Visualizing the graph via Networkx
A_ = Aq.abs()
xi, yi = np.where(A_)
edgelist = [
    (xi[n], yi[n], {'weight': A_[xi[n], yi[n]]})
    for n in range(len(xi))
]

g = nx.DiGraph()
g.add_edges_from(edgelist)
plot_graph(
    g, coords, figsize=(3, 3), node_size=60,
    edge_color=(0.8, 0.8, 0.8, 0.8))

# Total variation of eifenvectors
qgft2 = QGFT(norm=1)
qgft2.fit(Aq)

plt.figure(figsize=(4, 2))
plt.scatter(np.real(qgft2.eigc), np.imag(qgft2.eigc), c=qgft2.tv_)
plt.colorbar()
plt.title("Total Variation of eigenvectors for each eigenvalue")
plt.xlabel("Real(eigvals)")
plt.ylabel("Imag(eigvals)")
plt.tight_layout()
plt.show()
