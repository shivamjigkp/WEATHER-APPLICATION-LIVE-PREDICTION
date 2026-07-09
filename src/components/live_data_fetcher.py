"""
live_data_fetcher.py
------------------------
Fetches REAL recent hourly weather history for a location using Open-Meteo
(https://open-meteo.com) -- completely free, no API key required.

This replaces the earlier "approximate lag features from current reading"
approach in the frontend. Now the backend fetches the actual past ~8 days
of hourly data for the requested city and computes real lag/rolling
features from it.
"""
import sys

import pandas as pd
import requests

from src.exception import CustomException
from src.logger import logging


def fetch_live_weather_history(lat: float, lon: float, past_days: int = 8) -> pd.DataFrame:
    """Fetch hourly temperature/humidity/pressure/wind/precipitation history.

    past_days must be >= 8 so that a full 168-hour (7 day) lookback is
    available for the temp_lag_168hr feature.

    Returns a DataFrame with the same columns as the synthetic dataset used
    for training: timestamp, temperature, humidity, pressure, wind_speed, rain
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,precipitation",
            "past_days": past_days,
            "forecast_days": 1,
            "timezone": "auto",
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        hourly = response.json()["hourly"]

        df = pd.DataFrame(
            {
                "timestamp": hourly["time"],
                "temperature": hourly["temperature_2m"],
                "humidity": hourly["relative_humidity_2m"],
                "pressure": hourly["pressure_msl"],
                "wind_speed": hourly["wind_speed_10m"],
                "rain": [1 if (p is not None and p > 0.1) else 0 for p in hourly["precipitation"]],
            }
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Open-Meteo returns a little bit of future forecast too (forecast_days=1)
        # -- keep only rows up to the current time so lag features are real, not forecasted.
        now = pd.Timestamp.now()
        df = df[df["timestamp"] <= now].reset_index(drop=True)

        logging.info(f"Fetched {len(df)} hours of live weather history for lat={lat}, lon={lon}")
        return df

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    # Quick manual test -- New Delhi coordinates
    df = fetch_live_weather_history(lat=28.6139, lon=77.2090)
    print(df.tail())
    print("Total hours fetched:", len(df))
