"""
data_ingestion.py
------------------
Common data source for BOTH the temperature model and the rain model.

Right now this generates a realistic synthetic hourly weather dataset
(2 years) so the whole project is runnable end-to-end without needing
an API key or a downloaded dataset.

Later: replace `generate_synthetic_weather_data()` with a function that
reads a real CSV (e.g. Kaggle "Rain in Australia") or calls a historical
weather API. As long as the output DataFrame has the same columns, nothing
downstream (transformation/training files) needs to change.
"""
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("data", "raw", "weather_raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def generate_synthetic_weather_data(self, start="2023-01-01", periods_days=730):
        """Creates hourly weather data with daily + yearly seasonality."""
        rng = pd.date_range(start=start, periods=periods_days * 24, freq="h")
        n = len(rng)

        day_of_year = rng.dayofyear.values
        hour = rng.hour.values

        # Yearly seasonality (hot in summer, cold in winter) + daily seasonality (warm afternoon, cool night)
        yearly_cycle = 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        daily_cycle = 5 * np.sin(2 * np.pi * (hour - 9) / 24)
        base_temp = 22 + yearly_cycle + daily_cycle
        noise = np.random.normal(0, 1.5, n)
        temperature = base_temp + noise

        # Humidity: inversely related to temperature, roughly
        humidity = np.clip(70 - (temperature - 22) * 1.2 + np.random.normal(0, 5, n), 20, 100)

        # Pressure: mostly stable with small random walk
        pressure = 1013 + np.cumsum(np.random.normal(0, 0.05, n))
        pressure = np.clip(pressure, 990, 1030)

        # Wind speed
        wind_speed = np.clip(np.random.gamma(2, 2, n), 0, 40)

        # Rain probability increases with humidity and decreases with pressure
        rain_prob = 1 / (1 + np.exp(-(0.08 * (humidity - 75) - 0.05 * (pressure - 1013))))
        rain = np.random.binomial(1, rain_prob)

        df = pd.DataFrame(
            {
                "timestamp": rng,
                "temperature": temperature.round(2),
                "humidity": humidity.round(1),
                "pressure": pressure.round(1),
                "wind_speed": wind_speed.round(1),
                "rain": rain,
            }
        )
        return df

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion")
        try:
            df = self.generate_synthetic_weather_data()

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info(f"Raw weather data saved at {self.ingestion_config.raw_data_path}")
            return self.ingestion_config.raw_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    path = obj.initiate_data_ingestion()
    print(f"Raw data generated at: {path}")
