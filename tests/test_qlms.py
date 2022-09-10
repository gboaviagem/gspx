""" Unit testing the LMS workflow."""

import numpy as np
from pyquaternion import Quaternion
from gspx.signals import QuaternionSignal
from gspx.datasets import uk_weather

from gspx.utils.display import plot_graph
from gspx.datasets import WeatherGraphData
from gspx.qgsp import create_quaternion_weights, QGFT, QMatrix
from gspx.adaptive import QLMS


def test_qgft():
    """Test a quaternion graph signal denoising with FIR filter via QLMS."""
    uk_data = WeatherGraphData()
    Ar, _ = uk_data.graph
    s = uk_data.signal

    df = uk_weather()

    Aq = create_quaternion_weights(
        Ar, df, icols=['humidity'], jcols=['temp'],
        kcols=['wind_speed'])

    qgft = QGFT()
    qgft.fit(Aq)

    # Heat kernel in all 4 quaternion dimensions
    k = 0.2
    ss = np.zeros(len(qgft.idx_freq))
    ss[qgft.idx_freq] = np.exp(-k * np.arange(len(qgft.idx_freq)))

    ss = QuaternionSignal.from_rectangular(
        np.hstack([ss[:, np.newaxis]] * 4)
    )

    rnd = np.random.default_rng(seed=42)
    err_amplitude = 0.15

    nn = QuaternionSignal.from_equal_dimensions(
        rnd.uniform(low=-err_amplitude, high=err_amplitude, size=len(ss))
    )

    s = qgft.inverse_transform(ss)

    # Ideal LPF
    h_ideal = np.zeros(len(qgft.idx_freq))
    bandwidth = int(len(qgft.idx_freq) / 5)
    h_ideal[qgft.idx_freq[:bandwidth]] = 1
    h_idealq = QuaternionSignal.from_rectangular(np.hstack((
        h_ideal[:, np.newaxis],
        np.zeros(len(qgft.idx_freq))[:, np.newaxis],
        np.zeros(len(qgft.idx_freq))[:, np.newaxis],
        np.zeros(len(qgft.idx_freq))[:, np.newaxis]
    )))

    X = QMatrix.vander(qgft.eigq, 7, increasing=True)
    y = h_idealq

    qlms = QLMS(alpha=[0.3])
    qlms.fit(X, y)
    assert 'result' in qlms.res_[qlms.best_lr_]

    h_opt = qlms.predict(X)
    h_opt = QuaternionSignal.from_samples(h_opt.matrix.ravel())

    # Ideal filter
    sn = qgft.inverse_transform(ss + nn)
    mse_prior = np.mean((s - sn).abs()**2)

    ssn_lpf = (ss + nn).hadamard(h_idealq)
    s_lpf = qgft.inverse_transform(ssn_lpf)
    mse_post = np.mean((s - s_lpf).abs()**2)

    # FIR filter
    sn = qgft.inverse_transform(ss + nn)

    ssn_lpf = (ss + nn).hadamard(h_opt)
    s_lpf = qgft.inverse_transform(ssn_lpf)
    mse_post = np.mean((s - s_lpf).abs()**2)
