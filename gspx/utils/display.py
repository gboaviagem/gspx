"""Utilities for visualization."""
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import networkx as nx

from gspx.utils.quaternion_matrix import explode_quaternions
from gspx.utils.graph import to_networkx


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


def plot_graph(
        G, coords, colors=None, figsize=(10, 10), colormap=None,
        **draw_kwargs):
    """Display a graph and possibly its graph signal.

    Parameters
    ----------
    G : np.ndarray or nx.Graph
    coords : np.ndarray, shape=(N, 2)
    colors : gspx.QuaternionSignal or array-like, default=None
        Array-like of rgba tuples.
    figsize : tuple, length=2, default=(10, 10)
        Matplotlib's figsize, provided as (width, height).
    **draw_kwargs : dict
        Keyword arguments passed to nx.draw(), such as:

            - node_size; default=30
            - edge_color; default=(0.8, 0.8, 0.8, 0.3)
            - with_labels; default=False

    """
    from gspx.signals import QuaternionSignal

    if isinstance(colors, np.ndarray) and len(colors.shape) == 1:
        # We create a pseudocolor signal
        cmap_name = 'viridis' if colormap is None else colormap
        norm_ = (colors - colors.min()) / (colors.max() - colors.min())
        colors = get_cmap(cmap_name)(norm_)

    if not isinstance(G, nx.Graph):
        G = to_networkx(G)

    params = dict(
        node_size=30,
        edge_color=(0.8, 0.8, 0.8, 1.0),
        with_labels=False
    )

    if colors is not None:
        if hasattr(colors, 'to_rgba'):
            node_color = [tuple(rgba) for rgba in colors.to_rgba()]
        else:
            node_color = colors
        params['node_color'] = node_color

    params.update(draw_kwargs)

    plt.figure(figsize=figsize)
    nx.draw(G, pos=coords, **params)
