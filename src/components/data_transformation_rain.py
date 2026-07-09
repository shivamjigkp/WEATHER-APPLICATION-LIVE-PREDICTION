"""
data_transformation_rain.py
------------------------------
Cleaning + feature engineering ONLY for the Rain Prediction (classification)
model. Target: will it rain in the next 24 hours (1) or not (0)?
"""
import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationRainConfig:
    processed_data_path: str = os.path.join("data", "processed", "rain_features.csv")


class DataTransformationRain:
    def __init__(self):
        self.config = DataTransformationRainConfig()

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").sort_index()

            df["month"] = df.index.month
            df["hour"] = df.index.hour

            # Rolling weather trend features (last 24hr)
            df["humidity_rolling_mean_24hr"] = df["humidity"].rolling(24).mean()
            df["pressure_rolling_mean_24hr"] = df["pressure"].rolling(24).mean()
            df["pressure_change_24hr"] = df["pressure"] - df["pressure"].shift(24)

            # Did it rain in the last 24hr? (rain often clusters)
            df["rained_last_24hr"] = df["rain"].rolling(24).max()

            # Target: will it rain in the next 24 hours (classification)
            df["target_rain_next_24hr"] = df["rain"].shift(-24)

            df = df.dropna()
            df["target_rain_next_24hr"] = df["target_rain_next_24hr"].astype(int)

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def build_latest_feature_row(self, df: pd.DataFrame) -> pd.DataFrame:
        """For LIVE prediction: given real recent hourly history, returns a
        single-row DataFrame of feature values for the most recent timestamp.
        No target column here (needs future data we don't have yet).
        """
        try:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").sort_index()

            df["month"] = df.index.month
            df["hour"] = df.index.hour

            df["humidity_rolling_mean_24hr"] = df["humidity"].rolling(24).mean()
            df["pressure_rolling_mean_24hr"] = df["pressure"].rolling(24).mean()
            df["pressure_change_24hr"] = df["pressure"] - df["pressure"].shift(24)
            df["rained_last_24hr"] = df["rain"].rolling(24).max()

            feature_columns = [
                "temperature",
                "humidity",
                "pressure",
                "wind_speed",
                "month",
                "hour",
                "humidity_rolling_mean_24hr",
                "pressure_rolling_mean_24hr",
                "pressure_change_24hr",
                "rained_last_24hr",
            ]

            latest_row = df.iloc[[-1]][feature_columns]

            if latest_row.isnull().values.any():
                raise ValueError(
                    "Not enough historical data to compute all features. "
                    "Need at least 24 hours of hourly history."
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

            logging.info(f"Rain features saved at {self.config.processed_data_path}")

            feature_columns = [
                "temperature",
                "humidity",
                "pressure",
                "wind_speed",
                "month",
                "hour",
                "humidity_rolling_mean_24hr",
                "pressure_rolling_mean_24hr",
                "pressure_change_24hr",
                "rained_last_24hr",
            ]
            target_column = "target_rain_next_24hr"

            X = featured_df[feature_columns]
            y = featured_df[target_column]

            return X, y

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion

    raw_path = DataIngestion().initiate_data_ingestion()
    X, y = DataTransformationRain().initiate_data_transformation(raw_path)
    print(X.shape, y.shape, "rain rate:", y.mean())
