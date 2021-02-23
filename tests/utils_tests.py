from TaxiFareModel.utils import haversine_vectorized, compute_rmse
from TaxiFareModel.data import get_data
import numpy as np


def test_haversine():
    df = get_data(nrows=1)
    assert round(haversine_vectorized(df)[0],
                 2) == 1.03, "Distance not right"


def test_rmse():
    y_true = np.array((34, 37, 44, 47, 48, 48, 46, 43, 32, 27, 26, 24))
    y_pred = np.array((37, 40, 46, 44, 46, 50, 45, 44, 34, 30, 22, 23))
    assert round(compute_rmse(y_pred, y_true),
                 2) == 2.43, "RMSE calculation is not right"
