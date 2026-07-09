"""
model_trainer_temperature.py
------------------------------
Trains the XGBoost Regressor for Temperature Forecasting.
Saved model -> artifacts/temperature_model.pkl
"""
import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerTemperatureConfig:
    trained_model_path: str = os.path.join("artifacts", "temperature_model.pkl")


class ModelTrainerTemperature:
    def __init__(self):
        self.config = ModelTrainerTemperatureConfig()

    def initiate_model_training(self, X, y):
        try:
            # Chronological split (NOT random) -- same reasoning as the
            # electricity demand forecasting project: future data must not
            # leak into training.
            split_idx = int(len(X) * 0.85)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            model = XGBRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=6,
                objective="reg:squarederror",
                early_stopping_rounds=50,
                eval_metric="rmse",
            )

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            preds = model.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))

            logging.info(f"Temperature model -> RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}")

            save_object(self.config.trained_model_path, model)

            return {"rmse": rmse, "mae": mae, "r2": r2}

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation_temperature import DataTransformationTemperature

    raw_path = DataIngestion().initiate_data_ingestion()
    X, y = DataTransformationTemperature().initiate_data_transformation(raw_path)
    metrics = ModelTrainerTemperature().initiate_model_training(X, y)
    print(metrics)
