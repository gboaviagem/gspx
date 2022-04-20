"""Some useful datasets."""
import pathlib
import pandas as pd


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
