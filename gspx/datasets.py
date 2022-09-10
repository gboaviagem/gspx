"""Some useful datasets."""
import pathlib
import pandas as pd
import numpy as np
from typing import Union
import json

from gspx.utils.graph import nearest_neighbors, describe
from gspx.signals import QuaternionSignal

PATH_PREFIX = pathlib.Path(__file__).parent


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
    fn = PATH_PREFIX / "resources/uk_weather_at_20Apr202213pm.gz"
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

        s = QuaternionSignal.from_rectangular(weather_data)
        return s


class SocialGraphData:
    """Create a graph and graph signal out of socioeconomic data.

    The information concerns socioeconomic data of US Counties, from the
    Continguous United States.

    The graph edge weights are real-valued, and the graph signal is
    quaternion-valued.

    Parameters
    ----------
    n_neighbors: int, default=4
        Number of neighbors used in the determination of the edge
        relationships.
    dec: float, default=0.3
        Decimation factor. It is the fraction of counties with
        longitude greater than -102 which is retained (the rest is
        ignored). The objective here is to make the graph sparser.

    Examples
    --------
    >>> # First three dataset rows
    >>> data = SocialGraphData()
    >>> print(data.data.head(3).T)
                                               0         1             2
    county_fips                            37179      1081         42065
    county_ascii                           Union       Lee     Jefferson
    lat                                  34.9884   32.6011       41.1282
    lng                                 -80.5307  -85.3555      -78.9994
    state                         North Carolina   Alabama  Pennsylvania
    pop2017                             231366.0  161604.0       43804.0
    median_age_2017                         38.0      31.0          43.9
    bachelors_2017                          34.0      34.9          15.8
    median_household_income_2017         70858.0   47564.0       45342.0
    unemployment_rate_2017                   4.0      3.91          5.61
    uninsured_2017                           9.9       8.7           8.6

    >>> desc = data.describe_data()
    >>> desc['age_under_5_2010']
    'Percent of population under 5 (2010).'

    >>> # Graph weighted adjacency matrix and nodes geographic coordinates
    >>> A, coords = data.graph
    >>> data.describe_graph()
    n_nodes: 1267
    n_edges: 2357
    n_self_loops: 0
    density: 0.002938862434555137
    is_connected: True
    n_connected_components: 1
    is_directed: False
    is_weighted: True

    >>> # Visualize the graph
    >>> from gspx.utils.display import plot_graph
    >>> plot_graph(
    ...     A, coords=coords,
    ...     figsize=(10, 5),
    ...     colormap='viridis',
    ...     node_size=10)

    """

    def __init__(self, n_neighbors: int = 4, dec: float = 0.3):
        """Construct."""
        self.n_neighbors = n_neighbors
        assert dec <= 1 and dec > 0, (
            "The decimation factor must be positive and not greater than 1."
        )
        self.dec = dec
        self.data_ = None
        self.A_ = None
        self.coords_ = None

    def describe_data(self):
        """Describe the variables in the socioeconomic dataset."""
        with open(
                PATH_PREFIX / "resources/county_data_description.json",
                "r") as f:
            d = json.load(f)
        return d

    @property
    def data(self):
        """Return the socioeconomic dataset."""
        if self.data_ is not None:
            return self.data_

        # Source: https://simplemaps.com/data/us-counties
        df = pd.read_csv(PATH_PREFIX / "resources/county_latlong.gz")

        # Pruning counties outside the Contiguous United States
        df = df[df['lat'] < 52]
        df = df[df['lng'] > -150]

        # Decimating the nodes in the denser part of the graph
        # (longitudes greater than -102.)
        idx = df[df['lng'] > -102].index.tolist()
        notidx = df[df['lng'] <= -102].index.tolist()
        rnd = np.random.RandomState(seed=42)
        new_idx = rnd.permutation(idx)[:int(self.dec * len(idx))]
        df = df.loc[list(new_idx) + notidx]

        # County social data. Source:
        # https://www.openintro.org/data/?data=county_complete
        df_data = pd.read_csv(PATH_PREFIX / "resources/county_data.gz")
        df_data = df_data[[
            'fips', 'state', 'pop2017', 'median_age_2017',
            'bachelors_2017', 'median_household_income_2017',
            'unemployment_rate_2017', 'uninsured_2017'
        ]]

        self.data_ = pd.merge(
            df[['county_fips', 'county_ascii', 'lat', 'lng']],
            df_data.rename(
                {'fips': 'county_fips'}, axis="columns", inplace=False),
            on='county_fips')
        return self.data_

    @property
    def graph(self, df: Union[pd.DataFrame, None] = None):
        """Create a real-weighted nearest neighbors graph."""
        if self.A_ is not None and self.coords_ is not None:
            return self.A_, self.coords_

        if df is None:
            df = self.data if self.data_ is None else self.data_
        else:
            assert set(['lng', 'lat']).issubset(df.columns), (
                "The provided dataframe must contain the latitude "
                "and longitude columns named as 'lng' and 'lat'."
            )
        self.coords_ = df[['lng', 'lat']].to_numpy()

        A = nearest_neighbors(
            self.coords_, n_neighbors=self.n_neighbors,
            algorithm='ball_tree', mode='distance').todense()
        self.A_ = A + A.T
        return self.A_, self.coords_

    def describe_graph(self, A: np.ndarray = None, return_dict: bool = False):
        """Describe the created graph."""
        if A is None:
            A = self.graph[0] if self.A_ is None else self.A_
        return describe(A, return_dict)

    @property
    def signal(self, verbose: bool = True):
        """Create the quaternion graph signal with socioeconomic data."""
        df = self.data if self.data_ is None else self.data_
        cols = [
            'bachelors_2017', 'median_household_income_2017',
            'unemployment_rate_2017', 'uninsured_2017'
        ]
        df_ = df[cols]
        signal_data = (
            (df_ - df_.min()) /
            (df_.max() - df_.min())
        ).to_numpy()

        s = QuaternionSignal.from_rectangular(signal_data)
        if verbose:
            print(
                "Created a quaternion signal with components holding "
                f"the following information: {cols}.")
        return s
