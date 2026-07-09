"""
predict_pipeline_temperature.py
----------------------------------
Loads the trained temperature model and returns a forecast.

`CustomDataTemperature` is the same pattern Krish Naik uses: it packages
raw input values (coming from the Flask API / frontend) into the exact
DataFrame shape the model expects.
"""
import os
import sys

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class CustomDataTemperature:
    """Wraps raw inputs (e.g. from an API request) into a model-ready row.

    Note: `temp_lag_24hr`, `temp_lag_168hr`, `temp_rolling_mean_24hr` and
    `temp_rolling_std_24hr` normally come from real historical data. In a
    live system, the Flask app would look these up from a small rolling
    history buffer/database instead of the caller supplying them by hand.
    """

    def __init__(
        self,
        temperature: float,
        humidity: float,
        pressure: float,
        wind_speed: float,
        hour: int,
        dayofweek: int,
        month: int,
        is_weekend: int,
        temp_lag_24hr: float,
        temp_lag_168hr: float,
        temp_rolling_mean_24hr: float,
        temp_rolling_std_24hr: float,
    ):
        self.temperature = temperature
        self.humidity = humidity
        self.pressure = pressure
        self.wind_speed = wind_speed
        self.hour = hour
        self.dayofweek = dayofweek
        self.month = month
        self.is_weekend = is_weekend
        self.temp_lag_24hr = temp_lag_24hr
        self.temp_lag_168hr = temp_lag_168hr
        self.temp_rolling_mean_24hr = temp_rolling_mean_24hr
        self.temp_rolling_std_24hr = temp_rolling_std_24hr

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.__dict__])


class PredictPipelineTemperature:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "temperature_model.pkl")

    def predict(self, features: pd.DataFrame) -> float:
        try:
            model = load_object(self.model_path)
            prediction = model.predict(features)
            logging.info(f"Temperature prediction: {prediction}")
            return float(prediction[0])

        except Exception as e:
            raise CustomException(e, sys)

    def predict_from_location(self, lat: float, lon: float) -> float:
        """Fetches REAL recent history for this location (Open-Meteo), builds
        real lag/rolling features from it, and predicts the next-24hr
        temperature. No approximated/fake features involved."""
        try:
            from src.components.data_transformation_temperature import (
                DataTransformationTemperature,
            )
            from src.components.live_data_fetcher import fetch_live_weather_history

            history_df = fetch_live_weather_history(lat, lon)
            features = DataTransformationTemperature().build_latest_feature_row(history_df)
            return self.predict(features)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    sample = CustomDataTemperature(
        temperature=28.5,
        humidity=60,
        pressure=1012,
        wind_speed=8,
        hour=15,
        dayofweek=2,
        month=7,
        is_weekend=0,
        temp_lag_24hr=27.8,
        temp_lag_168hr=26.9,
        temp_rolling_mean_24hr=27.5,
        temp_rolling_std_24hr=1.8,
    )
    result = PredictPipelineTemperature().predict(sample.to_dataframe())
    print("Predicted temperature (next 24hr):", result)
