"""Some useful datasets."""
import pathlib
import pandas as pd
import numpy as np
from typing import Union
import json

from gspx.utils.graph import nearest_neighbors, describe
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

        s = QuaternionSignal.from_rectangular(weather_data)
        return s


class SocialGraphData:
    """Create a graph and graph signal out of socioeconomic data.

    The information concerns socioeconomic data of US Counties, from the
    Continguous United States.

    The graph edge weights are real-valued, and the graph signal is
    quaternion-valued.

    Examples
    --------
    >>> # First three dataset rows
    >>> data = SocialGraphData()
    >>> print(data.data.head(3).T)
                                            0          1          2
    county_fips                          6037      17031      48201
    county_ascii                  Los Angeles       Cook     Harris
    lat                               34.3209    41.8401    29.8578
    lng                             -118.2247   -87.8168   -95.3936
    state                          California   Illinois      Texas
    pop2017                        10163507.0  5211263.0  4652980.0
    median_age_2017                      36.0       36.4       33.1
    bachelors_2017                       31.2       37.2       30.5
    median_household_income_2017      61015.0    59426.0    57791.0
    unemployment_rate_2017               4.69       5.24        5.0
    uninsured_2017                       13.3       11.1       21.2

    >>> data.describe_data()
    {'state': 'State.',
    'name': 'County name.',
    'fips': 'FIPS code.',
    'pop2000': '2000 population.',
    'pop2010': '2010 population.',
    'pop2011': '2011 population.',
    'pop2012': '2012 population.',
    'pop2013': '2013 population.',
    'pop2014': '2014 population.',
    'pop2015': '2015 population.',
    'pop2016': '2016 population.',
    'pop2017': '2017 population.',
    'age_under_5_2010': 'Percent of population under 5 (2010).',
    'age_under_5_2017': 'Percent of population under 5 (2017).',
    'age_under_18_2010': 'Percent of population under 18 (2010).',
    'age_over_65_2010': 'Percent of population over 65 (2010).',
    'age_over_65_2017': 'Percent of population over 65 (2017).',
    'median_age_2017': 'Median age (2017).',
    'female_2010': 'Percent of population that is female (2010).',
    'white_2010': 'Percent of population that is white (2010).',
    'black_2010': 'Percent of population that is black (2010).',
    'black_2017': 'Percent of population that is black (2017).',
    'native_2010': 'Percent of population that is a Native American (2010).',
    'native_2017': 'Percent of population that is a Native American (2017).',
    'asian_2010': 'Percent of population that is a Asian (2010).',
    ...
    'uninsured_under_19_2019': (
        'Percent of population under 19 that is uninsured (2015-2019).'),
    'uninsured_65_and_older_2019': (
        'Percent of population 65 and older that is uninsured (2015-2019).'),
    'household_has_computer_2019': (
        'Percent of households that have desktop '
        'or laptop computer (2015-2019).'),
    'household_has_smartphone_2019': (
        'Percent of households that have smartphone (2015-2019).'),
    'household_has_broadband_2019': (
        'Percent of households that have broadband internet '
        'subscription (2015-2019).')}

    >>> # Graph weighted adjacency matrix and nodes geographic coordinates
    >>> A, coords = data.graph
    >>> data.describe_graph()
    n_nodes: 3108
    n_edges: 3826
    n_self_loops: 0
    density: 0.0007924150183564409
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

    def __init__(self, n_neighbors=3):
        """Construct."""
        self.n_neighbors = n_neighbors
        self.data_ = None
        self.A_ = None
        self.coords_ = None

    def describe_data(self):
        """Describe the variables in the socioeconomic dataset."""
        path_prefix = pathlib.Path(__file__).parent.parent
        with open(
                path_prefix / "resources/county_data_description.json",
                "r") as f:
            d = json.load(f)
        return d

    @property
    def data(self):
        """Return the socioeconomic dataset."""
        if self.data_ is not None:
            return self.data_

        # Source: https://simplemaps.com/data/us-counties
        path_prefix = pathlib.Path(__file__).parent.parent
        df = pd.read_csv(path_prefix / "resources/county_latlong.gz")

        # Pruning counties outside the Contiguous United States
        df = df[df['lat'] < 52]
        df = df[df['lng'] > -150]

        # County social data. Source:
        # https://www.openintro.org/data/?data=county_complete
        df_data = pd.read_csv(path_prefix / "resources/county_data.gz")
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
