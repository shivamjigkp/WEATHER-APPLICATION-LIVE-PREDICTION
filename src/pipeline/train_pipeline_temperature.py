"""
train_pipeline_temperature.py
--------------------------------
End-to-end: ingestion -> transformation -> training, for the TEMPERATURE
forecasting model.

Run with: python src/pipeline/train_pipeline_temperature.py
"""
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation_temperature import DataTransformationTemperature
from src.components.model_trainer_temperature import ModelTrainerTemperature
from src.exception import CustomException
from src.logger import logging


def run_temperature_training_pipeline():
    try:
        logging.info("===== Temperature Forecasting Training Pipeline started =====")

        raw_data_path = DataIngestion().initiate_data_ingestion()

        X, y = DataTransformationTemperature().initiate_data_transformation(raw_data_path)

        metrics = ModelTrainerTemperature().initiate_model_training(X, y)

        logging.info(f"===== Temperature Training Pipeline completed. Metrics: {metrics} =====")
        return metrics

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    result = run_temperature_training_pipeline()
    print("Temperature model trained. Metrics:", result)
