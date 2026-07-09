"""
predict_pipeline_rain.py
---------------------------
Loads the trained rain-prediction model and returns rain probability
for the next 24 hours.
"""
import os
import sys

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class CustomDataRain:
    def __init__(
        self,
        temperature: float,
        humidity: float,
        pressure: float,
        wind_speed: float,
        month: int,
        hour: int,
        humidity_rolling_mean_24hr: float,
        pressure_rolling_mean_24hr: float,
        pressure_change_24hr: float,
        rained_last_24hr: int,
    ):
        self.temperature = temperature
        self.humidity = humidity
        self.pressure = pressure
        self.wind_speed = wind_speed
        self.month = month
        self.hour = hour
        self.humidity_rolling_mean_24hr = humidity_rolling_mean_24hr
        self.pressure_rolling_mean_24hr = pressure_rolling_mean_24hr
        self.pressure_change_24hr = pressure_change_24hr
        self.rained_last_24hr = rained_last_24hr

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.__dict__])


class PredictPipelineRain:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "rain_model.pkl")

    def predict(self, features: pd.DataFrame):
        try:
            model = load_object(self.model_path)
            prediction = model.predict(features)
            probability = model.predict_proba(features)[:, 1]

            logging.info(f"Rain prediction: {prediction}, probability: {probability}")
            return int(prediction[0]), float(probability[0])

        except Exception as e:
            raise CustomException(e, sys)

    def predict_from_location(self, lat: float, lon: float):
        """Fetches REAL recent history for this location (Open-Meteo), builds
        real rolling features from it, and predicts rain for the next 24hr."""
        try:
            from src.components.data_transformation_rain import DataTransformationRain
            from src.components.live_data_fetcher import fetch_live_weather_history

            history_df = fetch_live_weather_history(lat, lon)
            features = DataTransformationRain().build_latest_feature_row(history_df)
            return self.predict(features)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    sample = CustomDataRain(
        temperature=24.0,
        humidity=85,
        pressure=1005,
        wind_speed=12,
        month=7,
        hour=18,
        humidity_rolling_mean_24hr=80,
        pressure_rolling_mean_24hr=1006,
        pressure_change_24hr=-3,
        rained_last_24hr=1,
    )
    will_rain, probability = PredictPipelineRain().predict(sample.to_dataframe())
    print("Will it rain in next 24hr?:", bool(will_rain), "| Probability:", round(probability, 3))
