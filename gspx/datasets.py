"""Some useful datasets."""
import pathlib
import pandas as pd
import numpy as np

from gspx.utils.graph import nearest_neighbors
from gspx.signals import QuaternionSignal


def uk_weather():
    """Fetch the dataframe with weather data in UK.

    All wind data was retrieved in 20 Apr 2022, at approximately 13:00 GMT.

    Source of the weather data: https://home.openweathermap.org/
    Source of the UK Towns lat-long data:
    https://www.latlong.net/category/towns-235-55.html

    Return
    ------
    df : pd.DataFrame

    """
    fn = (
        pathlib.Path(__file__).parent.parent /
        "resources/uk_weather_at_20Apr202213pm.gz")
    return pd.read_csv(fn, sep="\t")


class WeatherGraphData:
    """Build the graph and signal from UK weather data."""

    @property
    def graph(self):
        """Create the graph of UK cities and weather signal."""
        df = uk_weather()
        positions = df[['longitude', 'latitude']].to_numpy()
        coords = df[['longitude', 'latitude']].to_numpy()
        A = nearest_neighbors(
            positions,
            n_neighbors=10).todense()
        A = np.array(A) + np.array(A).T
        return A, coords

    @property
    def signal(self):
        """Create the graph signal with weather data."""
        df = uk_weather()
        df_ = df[['humidity', 'pressure', 'temp', 'wind_speed']]
        weather_data = (
            (df_ - df_.min()) /
            (df_.max() - df_.min())
        ).to_numpy()

        s = QuaternionSignal([
            dict(array=row) for row in weather_data
        ])
        return s
