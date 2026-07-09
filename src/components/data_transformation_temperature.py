"""
data_transformation_temperature.py
------------------------------------
Cleaning + feature engineering ONLY for the Temperature Forecasting
(regression) model. Builds lag features + rolling features, same technique
used in the earlier electricity demand forecasting project.
"""
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationTemperatureConfig:
    processed_data_path: str = os.path.join("data", "processed", "temperature_features.csv")


class DataTransformationTemperature:
    def __init__(self):
        self.config = DataTransformationTemperatureConfig()

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").sort_index()

            # Calendar features
            df["hour"] = df.index.hour
            df["dayofweek"] = df.index.dayofweek
            df["month"] = df.index.month
            df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

            # Lag features (most powerful predictors, same idea as demand forecasting)
            df["temp_lag_24hr"] = df["temperature"].shift(24)
            df["temp_lag_168hr"] = df["temperature"].shift(168)

            # Rolling features
            df["temp_rolling_mean_24hr"] = df["temperature"].rolling(24).mean()
            df["temp_rolling_std_24hr"] = df["temperature"].rolling(24).std()

            # Target: temperature 24 hours ahead (what we want to forecast)
            df["target_temp_next_24hr"] = df["temperature"].shift(-24)

            df = df.dropna()
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def build_latest_feature_row(self, df: pd.DataFrame) -> pd.DataFrame:
        """For LIVE prediction (not training): given real recent hourly
        history (fetched from live_data_fetcher), returns a single-row
        DataFrame of feature values for the most recent timestamp -- ready
        to feed into the trained model to forecast the next 24 hours from now.

        Unlike build_features(), this does NOT create the target column
        (which would need future data we don't have yet for live rows).
        """
        try:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").sort_index()

            df["hour"] = df.index.hour
            df["dayofweek"] = df.index.dayofweek
            df["month"] = df.index.month
            df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

            df["temp_lag_24hr"] = df["temperature"].shift(24)
            df["temp_lag_168hr"] = df["temperature"].shift(168)
            df["temp_rolling_mean_24hr"] = df["temperature"].rolling(24).mean()
            df["temp_rolling_std_24hr"] = df["temperature"].rolling(24).std()

            feature_columns = [
                "temperature",
                "humidity",
                "pressure",
                "wind_speed",
                "hour",
                "dayofweek",
                "month",
                "is_weekend",
                "temp_lag_24hr",
                "temp_lag_168hr",
                "temp_rolling_mean_24hr",
                "temp_rolling_std_24hr",
            ]

            latest_row = df.iloc[[-1]][feature_columns]

            if latest_row.isnull().values.any():
                raise ValueError(
                    "Not enough historical data to compute all features. "
                    "Need at least 7 days (168 hours) of hourly history."
                )

            return latest_row

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, raw_data_path: str):
        try:
            df = pd.read_csv(raw_data_path)
            featured_df = self.build_features(df)

            os.makedirs(os.path.dirname(self.config.processed_data_path), exist_ok=True)
            featured_df.to_csv(self.config.processed_data_path)

            logging.info(f"Temperature features saved at {self.config.processed_data_path}")

            feature_columns = [
                "temperature",
                "humidity",
                "pressure",
                "wind_speed",
                "hour",
                "dayofweek",
                "month",
                "is_weekend",
                "temp_lag_24hr",
                "temp_lag_168hr",
                "temp_rolling_mean_24hr",
                "temp_rolling_std_24hr",
            ]
            target_column = "target_temp_next_24hr"

            X = featured_df[feature_columns]
            y = featured_df[target_column]

            return X, y

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion

    raw_path = DataIngestion().initiate_data_ingestion()
    X, y = DataTransformationTemperature().initiate_data_transformation(raw_path)
    print(X.shape, y.shape)
