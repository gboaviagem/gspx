"""Unit testing manipulation of QuaternionSignal."""
import pytest
from pyquaternion import Quaternion

from gspx.signals import QuaternionSignal
from gspx.datasets import uk_weather


@pytest.fixture
def arr():
    """Input data."""
    arrays = [
        [2, 2, 8, 0],
        [4, 5, 6, 5],
        [1, 9, 4, 1],
        [0, 7, 8, 5],
        [6, 0, 4, 9],
        [7, 0, 9, 6],
        [9, 2, 7, 3],
        [9, 0, 0, 0],
        [9, 5, 8, 6],
        [0, 7, 8, 7]]
    return arrays


def test_from_rectangular(arr):
    """Test creation from rectangular form."""
    s = QuaternionSignal.from_rectangular(arr)
    assert isinstance(s, QuaternionSignal)
    expected = [
        Quaternion(2.0, 2.0, 8.0, 0.0),
        Quaternion(4.0, 5.0, 6.0, 5.0),
        Quaternion(1.0, 9.0, 4.0, 1.0),
        Quaternion(0.0, 7.0, 8.0, 5.0),
        Quaternion(6.0, 0.0, 4.0, 9.0),
        Quaternion(7.0, 0.0, 9.0, 6.0),
        Quaternion(9.0, 2.0, 7.0, 3.0),
        Quaternion(9.0, 0.0, 0.0, 0.0),
        Quaternion(9.0, 5.0, 8.0, 6.0),
        Quaternion(0.0, 7.0, 8.0, 7.0)]
    assert set(s.matrix.ravel().tolist()) == set(expected)


@pytest.mark.parametrize("axis,out_samples", [
    ("i", [
        Quaternion(2.0, 2.0, -8.0, 0.0),
        Quaternion(4.0, 5.0, -6.0, -5.0),
        Quaternion(1.0, 9.0, -4.0, -1.0),
        Quaternion(0.0, 7.0, -8.0, -5.0),
        Quaternion(6.0, 0.0, -4.0, -9.0),
        Quaternion(7.0, 0.0, -9.0, -6.0),
        Quaternion(9.0, 2.0, -7.0, -3.0),
        Quaternion(9.0, 0.0, 0.0, 0.0),
        Quaternion(9.0, 5.0, -8.0, -6.0),
        Quaternion(0.0, 7.0, -8.0, -7.0)
    ]),
    ("j", [
        Quaternion(2.0, -2.0, 8.0, 0.0),
        Quaternion(4.0, -5.0, 6.0, -5.0),
        Quaternion(1.0, -9.0, 4.0, -1.0),
        Quaternion(0.0, -7.0, 8.0, -5.0),
        Quaternion(6.0, 0.0, 4.0, -9.0),
        Quaternion(7.0, 0.0, 9.0, -6.0),
        Quaternion(9.0, -2.0, 7.0, -3.0),
        Quaternion(9.0, 0.0, 0.0, 0.0),
        Quaternion(9.0, -5.0, 8.0, -6.0),
        Quaternion(0.0, -7.0, 8.0, -7.0)
    ]),
    ("k", [
        Quaternion(2.0, -2.0, -8.0, 0.0),
        Quaternion(4.0, -5.0, -6.0, 5.0),
        Quaternion(1.0, -9.0, -4.0, 1.0),
        Quaternion(0.0, -7.0, -8.0, 5.0),
        Quaternion(6.0, 0.0, -4.0, 9.0),
        Quaternion(7.0, 0.0, -9.0, 6.0),
        Quaternion(9.0, -2.0, -7.0, 3.0),
        Quaternion(9.0, 0.0, 0.0, 0.0),
        Quaternion(9.0, -5.0, -8.0, 6.0),
        Quaternion(0.0, -7.0, -8.0, 7.0)
    ]),
    ([0.3, 1.3, 0.4, 1.0], [
        Quaternion(3.0612244897959178, 2.5986394557823127,
                   -6.5850340136054415, 3.5374149659863945),
        Quaternion(6.591836734693876, 6.231292517006802,
                   -2.5442176870748283, 3.639455782312925),
        Quaternion(3.857142857142856, 3.3809523809523796,
                   -0.19047619047619035, 8.523809523809522),
        Quaternion(3.5306122448979584, 8.299319727891156,
                   -3.2925170068027194, 6.768707482993197),
        Quaternion(7.795918367346937, 7.782312925170067,
                   -1.6054421768707487, -3.013605442176869),
        Quaternion(8.530612244897958, 6.632653061224488,
                   -6.9591836734693855, -0.8979591836734679),
        Quaternion(10.163265306122447, 3.040816326530612,
                   -5.448979591836733, 0.8775510204081636),
        Quaternion(8.448979591836732, -2.387755102040815,
                   -0.7346938775510208, -1.8367346938775504),
        Quaternion(11.653061224489795, 6.496598639455781,
                   -4.462585034013604, 2.8435374149659856),
        Quaternion(3.938775510204081, 10.06802721088435,
                   -2.7482993197278907, 6.129251700680271)])
])
def test_involution(arr, axis, out_samples):
    """Test the involution method."""
    s = QuaternionSignal.from_rectangular(arr)
    q = s.involution(axis, inplace=False)

    assert set(q.matrix.ravel().tolist()) == set(out_samples)

    s.involution(axis, inplace=True)
    assert set(s.matrix.ravel().tolist()) == set(out_samples)


def test_uk_weather():
    """Test the dataset."""
    df = uk_weather()
    s = QuaternionSignal.from_rectangular(
        df[['wind_speed', 'humidity', 'pressure', 'temp']].to_numpy()
    )
    assert s.shape == (177, 1)
