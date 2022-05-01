import pytest
from gspx.signals import QuaternionSignal
from pyquaternion import Quaternion


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
    assert set(s.samples.tolist()) == set(expected)
