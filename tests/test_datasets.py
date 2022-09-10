""" Unit testing the datasets."""

from gspx.utils.display import plot_graph
from gspx.datasets import SocialGraphData


def test_social():
    data = SocialGraphData()
    assert set(data.data.columns) == set([
        'county_fips', 'county_ascii', 'lat', 'lng', 'state', 'pop2017',
        'median_age_2017', 'bachelors_2017', 'median_household_income_2017',
        'unemployment_rate_2017', 'uninsured_2017'
    ])
    assert data.describe_graph(return_dict=True) == {
        'n_nodes': 1267,
        'n_edges': 2357,
        'n_self_loops': 0,
        'density': 0.002938862434555137,
        'is_connected': True,
        'n_connected_components': 1,
        'is_directed': False,
        'is_weighted': True
    }
    A, coords = data.graph
    plot_graph(A, coords)
