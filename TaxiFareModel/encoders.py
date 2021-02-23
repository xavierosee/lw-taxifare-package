from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from TaxiFareModel.utils import haversine_vectorized


class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """Extract the day of week (dow), the hour, the month and the year from a
    time column."""

    def __init__(self, time_column, time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only four columns:
        'dow', 'hour', 'month', 'year'"""
        _X = X.copy()
        _X.index = pd.to_datetime(X[self.time_column])
        _X.index = _X.index.tz_convert(self.time_zone_name)
        _X["dow"] = _X.index.weekday
        _X["hour"] = _X.index.hour
        _X["month"] = _X.index.month
        _X["year"] = _X.index.year
        return _X[['dow', 'hour', 'month', 'year']]


class DistanceTransformer(BaseEstimator, TransformerMixin):
    """Compute the haversine distance between two GPS points."""

    def __init__(self,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only 1 column: 'distance'"""
        df = pd.DataFrame(haversine_vectorized(X.copy()))
        df.columns = ['distance']
        return df
