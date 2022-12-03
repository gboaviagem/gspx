"""Some useful datasets."""
from abc import abstractmethod
import pathlib
import pandas as pd
import numpy as np
from typing import Union
import json

from gspx.utils.graph import nearest_neighbors, describe
from gspx.utils.quaternion_matrix import explode_quaternions
from gspx.signals import QuaternionSignal
from gspx.qgsp.utils import create_quaternion_weights

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


class BaseGraphData:
    """Base class for GraphData classes."""

    def __init__(self, n_neighbors, feature_names, verbose: bool = True):
        """Construct."""
        self.n_neighbors = n_neighbors
        self.verbose = verbose
        self.feature_names = feature_names
        self.A_ = None
        self.data_ = None
        self.coords_ = None

    @property
    @abstractmethod
    def data(self):
        """Return the dataframe with tabular data related to the graph."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def graph(self):
        """Return the real-valued adjacency matrix and node coordinates."""
        raise NotImplementedError()

    def quaternion_adjacency_matrix(
            self, gauss_den: float = 2, hermitian: bool = True, **kwargs):
        """Return the quaternion-valued adjacency matrix.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe with the 
        gauss_den : float, default=2
            Integer assigned to the denominator in the gaussian
            weight distribution, as in `exp(- (x) / gauss_den)`.
            It is related to the gaussian standard deviation.
        hermitian : bool, default=True
            If True, make the quaternion output matrix hermitian.

        """
        Ar, _ = self.graph
        features = explode_quaternions(self.signal.matrix)[:, 0, :]
        df = pd.DataFrame(features, columns=self.feature_names)
        Aq = create_quaternion_weights(
            Ar, df, cols1 = [self.feature_names[0]],
            icols=[self.feature_names[1]], jcols=[self.feature_names[2]],
            kcols=[self.feature_names[3]], gauss_den=gauss_den,
            hermitian=hermitian, **kwargs)
        return Aq

    def describe_graph(self, return_dict: bool = False):
        """Describe the created graph."""
        A = self.graph[0] if self.A_ is None else self.A_
        return describe(A, return_dict)


class WeatherGraphData(BaseGraphData):
    """Build the graph and signal from UK weather data.

    Parameters
    ----------
    n_neighbors: int, default=4
        Number of neighbors used in the determination of the edge
        relationships.
    england_only: bool, default=True
        If True, then only the towns in England are used.

    Examples
    --------
    >>> data = WeatherGraphData(n_neighbors=7, england_only=True)
    >>> print(data.data.head(1).T)
                                        1
    town          St.Asaph, Wales, the UK
    latitude                    53.257999
    longitude                      -3.442
    humidity                           59
    pressure                         1017
    temp                           288.49
    wind_speed                       3.71
    wind_degrees                       99

    >>> s = data.signal
    >>> s.to_array()[:5, :]
    array([[0.54545455, 0.32352941, 0.66127168, 0.37003405],
          [0.09090909, 0.23529412, 0.84624277, 0.8830874 ],
          [0.18181818, 0.29411765, 0.84393064, 0.70828604],
          [1.        , 0.41176471, 0.        , 0.41089671],
          [0.40909091, 0.32352941, 0.81040462, 0.30419977]])

    >>> data = WeatherGraphData(n_neighbors=7, england_only=True)
    >>> data.describe_graph()
    n_nodes: 145
    n_edges: 542
    n_self_loops: 0
    density: 0.051915708812260535
    is_connected: True
    n_connected_components: 1
    is_directed: False
    is_weighted: True

    >>> data = WeatherGraphData(n_neighbors=7, england_only=False)
    >>> data.describe_graph()
    n_nodes: 177
    n_edges: 670
    n_self_loops: 0
    density: 0.043014894709809966
    is_connected: False
    n_connected_components: 3
    is_directed: False
    is_weighted: True

    """

    def __init__(self, n_neighbors: int = 4, england_only: bool = True):
        """Construct."""
        feature_names = ['humidity', 'pressure', 'temp', 'wind_speed']
        BaseGraphData.__init__(
            self, n_neighbors=n_neighbors, feature_names=feature_names)
        self.england_only = england_only

    @property
    def data(self):
        """Return the UK weather dataset."""
        if self.data_ is not None:
            return self.data_
        df = uk_weather()
        if self.england_only:
            # Trying to extract only towns in England.
            # First removing towns with latitude greater than 55.423
            df = df[df['latitude'] <= 55.4223]
            df = df[df['longitude'] >= -5.476]
        self.data_ = df
        return df

    @property
    def graph(self):
        """Create the graph of UK cities and weather signal."""
        if self.A_ is not None and self.coords_ is not None:
            return self.A_, self.coords_
        df = self.data
        positions = df[['longitude', 'latitude']].to_numpy()
        self.coords_ = df[['longitude', 'latitude']].to_numpy()
        A = nearest_neighbors(
            positions,
            n_neighbors=self.n_neighbors).todense()
        self.A_ = np.array(A) + np.array(A).T
        return self.A_, self.coords_

    @property
    def signal(self):
        """Create the graph signal with weather data."""
        df = self.data
        df_ = df[self.feature_names]
        weather_data = (
            (df_ - df_.min()) /
            (df_.max() - df_.min())
        ).to_numpy()

        s = QuaternionSignal.from_rectangular(weather_data)
        return s


class SocialGraphData(BaseGraphData):
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

    def __init__(
            self, n_neighbors: int = 4, dec: float = 0.3,
            verbose: bool = True):
        """Construct."""
        feature_names = [
            'bachelors_2017', 'median_household_income_2017',
            'unemployment_rate_2017', 'uninsured_2017'
        ]
        BaseGraphData.__init__(
            self, n_neighbors=n_neighbors, verbose=verbose,
            feature_names=feature_names)
        assert dec <= 1 and dec > 0, (
            "The decimation factor must be positive and not greater than 1."
        )
        self.dec = dec

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

    @property
    def signal(self):
        """Create the quaternion graph signal with socioeconomic data."""
        df = self.data if self.data_ is None else self.data_
        df_ = df[self.feature_names]
        signal_data = (
            (df_ - df_.min()) /
            (df_.max() - df_.min())
        ).to_numpy()

        s = QuaternionSignal.from_rectangular(signal_data)
        if self.verbose:
            print(
                "Created a quaternion signal with components holding "
                f"the following information: {self.feature_names}.")
        return s
