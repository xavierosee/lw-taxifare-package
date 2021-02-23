from TaxiFareModel.trainer import Trainer
from TaxiFareModel.data import get_data
import pandas as pd


def test_pipeline():
    df = get_data(nrows=1)
    X = df.drop(columns='fare_amount')
    y = df.fare_amount
    trainer = Trainer(X, y)
    trainer.set_pipeline()
    assert len(trainer.pipeline.get_params()['steps']) == 2


def test_fit():
    df = get_data(nrows=50)
    X = df.drop(columns='fare_amount')
    y = df.fare_amount
    trainer = Trainer(X, y)
    trainer.run()
    assert len(trainer.pipeline.get_params()['steps']) == 2
