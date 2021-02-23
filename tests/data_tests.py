from TaxiFareModel.data import get_data, clean_data


def test_number_of_columns():
    assert get_data(nrows=5).shape == (5, 8)


def test_cleaned_data():
    df = get_data(nrows=100)
    assert clean_data(df).shape[0] <= df.shape[0]
