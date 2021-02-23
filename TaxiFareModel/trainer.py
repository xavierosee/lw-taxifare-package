import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
import joblib

MLFLOW_URI = "https://mlflow.lewagon.co/"
myname = "xavierosee"
EXPERIMENT_NAME = f"[BE][Brussels][{myname}][TaxiFareModel]"


class Trainer(object):
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = EXPERIMENT_NAME

    def set_experiment_name(self, experiment_name):
        '''defines the experiment name for MLFlow'''
        self.experiment_name = experiment_name

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_cols = ['pickup_latitude', 'pickup_longitude',
                     'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']

        pipe_distance = make_pipeline(DistanceTransformer(), RobustScaler())
        pipe_time = make_pipeline(TimeFeaturesEncoder(
            'pickup_datetime'), OneHotEncoder(handle_unknown='ignore'))

        preproc = make_column_transformer((pipe_distance, dist_cols),
                                          (pipe_time, time_cols))

        self.pipeline = make_pipeline(preproc,
                                      RandomForestRegressor())

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.mlflow_log_param("model", "Random Forest Regressor")
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = round(compute_rmse(y_pred, y_test), 2)
        self.mlflow_log_metric("rmse", rmse)
        return rmse

    # MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, 'model.joblib')


if __name__ == "__main__":
    N = 10_000
    df = get_data(nrows=N)
    df = clean_data(df)
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    trainer = Trainer(X_train, y_train)
    trainer.run()
    trainer.save_model()
    rmse = trainer.evaluate(X_test, y_test)
    print(f"rmse: {rmse}")
