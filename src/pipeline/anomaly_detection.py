"""
anomaly_detection.py
------------------------
Weather Anomaly Detection: is the CURRENT weather reading unusual compared
to the historical distribution for this time of year?

Uses simple, explainable statistics (mean + std per month) rather than a
black-box model -- easy to explain in an interview: "z-score based anomaly
detection", and can be swapped for IsolationForest later if you want a
fancier answer.
"""
import sys

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging


class AnomalyDetector:
    def __init__(self, raw_data_path: str = "data/raw/weather_raw.csv"):
        self.raw_data_path = raw_data_path

    def _monthly_stats(self) -> pd.DataFrame:
        df = pd.read_csv(self.raw_data_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["month"] = df["timestamp"].dt.month
        stats = df.groupby("month")["temperature"].agg(["mean", "std"]).reset_index()
        return stats

    def check(self, current_temperature: float, current_month: int, z_threshold: float = 2.0):
        try:
            stats = self._monthly_stats()
            row = stats[stats["month"] == current_month].iloc[0]
            mean, std = row["mean"], row["std"]

            z_score = (current_temperature - mean) / std if std > 0 else 0
            is_anomaly = bool(abs(z_score) > z_threshold)

            if is_anomaly:
                message = (
                    "Unusually hot for this time of year"
                    if z_score > 0
                    else "Unusually cold for this time of year"
                )
            else:
                message = "Normal weather for this time of year"

            logging.info(
                f"Anomaly check -> temp={current_temperature}, month={current_month}, "
                f"z={z_score:.2f}, anomaly={is_anomaly}"
            )

            return {
                "is_anomaly": is_anomaly,
                "z_score": round(float(z_score), 2),
                "expected_mean": round(float(mean), 2),
                "message": message,
            }

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    result = AnomalyDetector().check(current_temperature=40, current_month=1)
    print(result)
