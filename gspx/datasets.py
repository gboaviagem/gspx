"""Some useful datasets."""
import pathlib
import pandas as pd


def uk_wind():
    """Fetch the dataframe with wind speed and direction in UK.

    All wind data was retrieved in 16 Apr 2022, at approximately 19:00 GMT.
    Source of the wind data: https://home.openweathermap.org/
    Source of the UK Towns lat-long data:
    https://www.latlong.net/category/towns-235-55.html

    """
    fn = (
        pathlib.Path(__file__).parent.parent /
        "resources/uk_towns_wind_at_16Apr202219pm.gz")
    return pd.read_csv(fn, sep="\t")


def uk_weather():
    """Fetch the dataframe with weather data in UK.

    All wind data was retrieved in 20 Apr 2022, at approximately 13:00 GMT.
    Source of the data: https://home.openweathermap.org/
    Source of the UK Towns lat-long data:
    https://www.latlong.net/category/towns-235-55.html

    """
    fn = (
        pathlib.Path(__file__).parent.parent /
        "resources/uk_weather_at_20Apr202213pm.gz")
    return pd.read_csv(fn, sep="\t")
