"""Basic spectral analysis using Quaternion Graph Signal Processing (QGSP)."""
import sys
import os
import pathlib
pdir = pathlib.Path(__file__).parent.resolve()

# Adding the gspx folder to the Python path
sys.path.insert(0, str(pdir).split("gspx")[0] + "/gspx")

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from gspx.datasets import WeatherGraphData
from gspx.signals import QuaternionSignal
from gspx.adaptive import QLMS
from gspx.qgsp import create_quaternion_weights, QGFT, QMatrix
from gspx.utils.display import plot_quaternion_graph_signal, plot_graph
from io import BytesIO


def show_and_save(
        plot_name: str, fig = None, save_to: str = "/figures",
        dpi: int = None, fig_size_inches: tuple = None,
        img_format: str = "pdf"):
    """Show the figure in Streamlit and saves locally."""
    if fig is None:
        fig = plt.gcf()
    if fig_size_inches is not None:
        fig.set_size_inches(fig_size_inches[0], fig_size_inches[1])
    if dpi is not None:
        fig.set_dpi(dpi)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)

    try:
        path = "./" + save_to.lstrip("/").rstrip("/")
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(f'{path}/{plot_name}.{img_format}', dpi=300)
    except Exception as e:
        print(f"Unable to save figure `{plot_name}`: ", e)


st.markdown(
    """
    #### ***QGSP in action***

    # Filter design using QLMS

    This streamlit app aims to

    - Create a graph with quaternion-valued edge weights.
    - Define a quaternion-valued graph signal.
    - Compute the Quaternion Graph Fourier Transform (QGFT) direct and inverse matrices.
    - Generate the quaternion graph signal spectrum.

    Let us first build a graph connecting some cities in England, using edges with quaternion-valued weights and define over it a quaternion-valued signal.
    """
)

data = WeatherGraphData(n_neighbors=7, england_only=True)
Ar, coords = data.graph
s = data.signal
df = data.data

st.markdown("The graph signal will have quaternion components (1, i, j, k) given by the columns `'humidity', 'pressure', 'temp', 'wind_speed'` in the following dataframe:")

st.dataframe(data=df, width=1000, height=None)

st.markdown("The quaternion adjacency matrix was created using `gspx` through the following code:")
st.code(
"""
from gspx.datasets import WeatherGraphData

data = WeatherGraphData(n_neighbors=7, england_only=True)
Ar, coords = data.graph
s = data.signal
df = data.data

Aq = data.quaternion_adjacency_matrix()
""",
language="python"
)

st.markdown("The graph is depicted in the figure below.")

plot_graph(
    Ar, coords, figsize=(4.5, 5.5))
show_and_save(
    plot_name="uk_graph",
    fig_size_inches=(3.5, 4.5))

st.markdown("The data in columns `'pressure', 'humidity', 'temp', 'wind_speed'` is normalized column-wise using linear scaling to make each column fit the range [0, 1]. The graph signal can be visualized by splitting each quaternion component in a pseudocolor scale as such:")

@st.cache
def get_quaternion_adj():
    """Run and cache."""
    Aq = data.quaternion_adjacency_matrix()
    return Aq

st.markdown("**Figure: Quaternion graph signal.**")
with st.spinner('Computing the quaternion-valued adjacency matrix.'):
    Aq = get_quaternion_adj()

fig = plot_quaternion_graph_signal(
    s, coords, figsize=(5, 8)
)
show_and_save(
    plot_name="uk_graph_signal", fig=fig,
    fig_size_inches=(4.5, 5.5))

st.markdown(
    """
    ### Quaternion Graph Fourier Transform (QGFT)
    The QGFT is computed by the `qgft = QGFT()` object. The eigenvalues and always complex-valued and are stored in the attribute `qgft.eigc`. The associated eigenvectors have their total variation stored in the attribute `qgft.tv_`.
    """
)

@st.cache
def compute_qgft(Aq):
    """Run and cache."""
    qgft = QGFT(norm=1)
    qgft.fit(Aq)
    return qgft
    
with st.spinner('Computing the QGFT.'):
    qgft = compute_qgft(Aq)

    plt.figure(figsize=(5, 5), dpi=100)
    plt.scatter(np.real(qgft.eigc), np.imag(qgft.eigc), c=qgft.tv_)
    plt.colorbar()
    plt.title("Total Variation of eigenvectors for each eigenvalue")
    plt.xlabel("Real(eigvals)")
    plt.ylabel("Imag(eigvals)")

st.markdown("Since this graph was created with a Hermitian adjacency matrix, its eigenvalues are always real-valued:")

show_and_save(
    plot_name="uk_total_variation",
    fig_size_inches=(6, 3.5))

st.markdown(
    """
    ### Signal spectrum

    The spectrum of the quaternion graph signal `s` is obtained by the method `qgft.transform(s)`. The following code illustrates is computation and the spectrum visual representation:
    """
)
st.code(
"""
from gspx.signals import QuaternionSignal

ss = qgft.transform(s)
QuaternionSignal.show(ss, ordering=qgft.idx_freq)
""",
language="python"
)

@st.cache
def qgft_transform(s):
    """Run and cache."""
    return qgft.transform(s)

with st.spinner('Transforming the input signal.'):
    ss = qgft_transform(s)

fig = QuaternionSignal.show(ss, ordering=qgft.idx_freq)

show_and_save(
    plot_name="uk_spectrum", fig=fig,
    fig_size_inches=(9, 5.5))

st.markdown(
    """
    ### Low-pass filter design using QLMS

    We will try to design a linear and shift-invariant (LSI) that approximates in the least-squares sense the following ideal filter:
    """
)

h_ideal = np.zeros(len(qgft.idx_freq))
# Bandwith of 20% the frequency support
bandwidth = int(len(qgft.idx_freq) / 5)
h_ideal[qgft.idx_freq[:bandwidth]] = 1

h_idealq = QuaternionSignal.from_rectangular(np.hstack((
    h_ideal[:, np.newaxis],
    np.zeros(len(qgft.idx_freq))[:, np.newaxis],
    np.zeros(len(qgft.idx_freq))[:, np.newaxis],
    np.zeros(len(qgft.idx_freq))[:, np.newaxis]
)))

fig = QuaternionSignal.show(h_idealq, ordering=qgft.idx_freq)
show_and_save(
    plot_name="uk_ideal_filter", fig=fig,
    fig_size_inches=(9, 5.5))

st.markdown(
    """
    Let us add some harsh white noise in the graph signal, what yields the following spectrum:
    """
)

ss = QuaternionSignal.from_samples(ss.matrix.ravel())
rnd = np.random.default_rng(seed=42)
err_amplitude = 5
nn = QuaternionSignal.from_equal_dimensions(
    rnd.uniform(low=-err_amplitude, high=err_amplitude, size=len(ss))
)

fig = QuaternionSignal.show(ss + nn, ordering=qgft.idx_freq)
show_and_save(
    plot_name="uk_spectrum_noisy", fig=fig,
    fig_size_inches=(9, 5.5))

st.markdown("The QLMS in now executed trying out the following values of step size: `[0.0001, 0.0005, 0.001, 0.002]`. The filter has **N = 7** filter taps (i.e. it is a polynomial of degree **L = 6** on the graph adjacency matrix).")

@st.cache
def qlms_computation():
    N = 7
    X = QMatrix.vander(qgft.eigq, N, increasing=True)
    y = h_idealq

    qlms = QLMS(
        step_size=[0.0001, 0.0005, 0.001, 0.002], verbose=2)
    qlms.fit(X, y)
    return X, qlms

with st.spinner('QLMS computation:'):
    X, qlms = qlms_computation()

st.markdown("The plot below shows the cost per iteration of the QLMS (only for the values of step sizes that did not cause interruption by divergence - our implementation has an `early stop` parameter that interrupts the calculations if the cost increases for more than 10 iterations).")
qlms.plot(nsamples=100)

show_and_save(
    plot_name="uk_qlms_iterations",
    fig_size_inches=(5.5, 4.5))

st.markdown("The filter taps obtained by the end of the optimization are:")
st.text(str(qlms.res_[qlms.best_lr_]['result']))

st.markdown(
    """
    ### LSI filter response

    The LSI filter obtained using QLMS has the following filter response:
    """
)

h_opt = qlms.predict(X)
h_opt = QuaternionSignal.from_samples(h_opt.matrix.ravel())
fig = QuaternionSignal.show(h_opt, ordering=qgft.idx_freq)
show_and_save(
    plot_name="uk_qlm_filter", fig=fig,
    fig_size_inches=(9, 5.5))

st.markdown(
    """
    ### Signal reconstruction using both the ideal and the FIR low-pass filters

    Let us now try to reconstruct the original signal out of the harshly
    noisy version.

    Firstly, let us use our **ideal LPF filter**:
    """
)

sn = qgft.inverse_transform(ss + nn)
def norm_mse(s_ref, s_approx):
    mse = np.mean((s_ref - s_approx).abs()**2)
    orig_mse = np.mean((s_ref).abs()**2)
    return mse / orig_mse

mse_ = norm_mse(s, sn)
st.markdown(
    f"Mean squared error (MSE) before filtering (noisy signal): {mse_}")

ssn_lpf = (ss + nn).hadamard(h_idealq)
s_lpf = qgft.inverse_transform(ssn_lpf)

mse_ = norm_mse(s, s_lpf)
st.markdown(
    f"Mean squared error (MSE) after filtering: {mse_}")

fig = QuaternionSignal.show(ssn_lpf, ordering=qgft.idx_freq)
show_and_save(
    plot_name="ideal_filter_output", fig=fig,
    fig_size_inches=(9, 5.5))

st.markdown("Now let us use our FIR LSI filter:.")

mse_ = norm_mse(s, sn)
st.markdown(
    f"Mean squared error (MSE) before filtering (noisy signal): {mse_}")

ssn_lpf = (ss + nn).hadamard(h_opt)
s_lpf = qgft.inverse_transform(ssn_lpf)

mse_ = norm_mse(s, s_lpf)
st.markdown(
    f"Mean squared error (MSE) after filtering: {mse_}")

fig = QuaternionSignal.show(ssn_lpf, ordering=qgft.idx_freq)
show_and_save(
    plot_name="qlms_filter_output", fig=fig,
    fig_size_inches=(9, 5.5))
