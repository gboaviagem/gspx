"""Utilities for visualization."""
import matplotlib.pyplot as plt
import numpy as np
from gspx.utils.quaternion_matrix import explode_quaternions


def visualize_quat_mtx(M, dpi=None):
    """Plot heatmap of quaternion matrix component-wise.

    Parameters
    ----------
    M : np.ndarray, shape=(N, M, 4)
    dpi : integer, optional, default: None
        Resolution of the figure. If not provided, defaults
        to rcParams["figure.dpi"] (default: 100.0) = 100.

    """
    assert isinstance(M, np.ndarray)
    if isinstance(M.dtype, type(np.dtype('O'))) and len(M.shape) == 2:
        mat = explode_quaternions(M)
    else:
        mat = M

    A = mat[:, :, 0]
    B = mat[:, :, 1]
    C = mat[:, :, 2]
    D = mat[:, :, 3]

    _, axs = plt.subplots(2, 2, dpi=dpi)
    axs[0, 0].imshow(A)
    axs[0, 0].set_title("Real part")
    axs[0, 1].imshow(B)
    axs[0, 1].set_title("i-component")
    axs[1, 0].imshow(C)
    axs[1, 0].set_title("j-component")
    axs[1, 1].imshow(D)
    axs[1, 1].set_title("k-component")
    plt.subplots_adjust(
        wspace=0.5, hspace=0.5)
    plt.show()
