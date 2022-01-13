"""Utilities for visualization."""
import matplotlib.pyplot as plt


def visualize_quat_mtx(M, dpi=None):
    """Plot heatmap of quaternion matrix component-wise.

    Parameters
    ----------
    M : np.ndarray, shape=(N, M, 4)
    dpi : integer, optional, default: None
        Resolution of the figure. If not provided, defaults
        to rcParams["figure.dpi"] (default: 100.0) = 100.

    """
    A = M[:, :, 0]
    B = M[:, :, 1]
    C = M[:, :, 2]
    D = M[:, :, 3]

    _, axs = plt.subplots(2, 2, dpi=dpi)
    axs[0, 0].imshow(A)
    axs[0, 0].set_title("Real part")
    axs[0, 1].imshow(B)
    axs[0, 0].set_title("i-component")
    axs[1, 0].imshow(C)
    axs[0, 0].set_title("j-component")
    axs[1, 1].imshow(D)
    axs[0, 0].set_title("k-component")
    plt.show()
