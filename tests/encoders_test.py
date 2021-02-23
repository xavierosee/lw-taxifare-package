from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.data import get_data


def test_time_encoder():
    df = get_data(nrows=1)
    X = df.drop(columns='fare_amount')
    y = df.fare_amount
    time_enc = TimeFeaturesEncoder('pickup_datetime')
    time_features = time_enc.fit_transform(X, y)
    assert time_features.shape[1] == 4, "shape[1] is not 4"


def test_distance_transformer():
    df = get_data(nrows=1)
    X = df.drop(columns='fare_amount')
    y = df.fare_amount
    dist_enc = DistanceTransformer()
    enc_features = dist_enc.fit_transform(X, y)
    assert enc_features.columns[0] == 'distance', "column name is not distance"
