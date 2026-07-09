"""
train_pipeline_rain.py
-------------------------
End-to-end: ingestion -> transformation -> training, for the RAIN
prediction model.

Run with: python src/pipeline/train_pipeline_rain.py
"""
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation_rain import DataTransformationRain
from src.components.model_trainer_rain import ModelTrainerRain
from src.exception import CustomException
from src.logger import logging


def run_rain_training_pipeline():
    try:
        logging.info("===== Rain Prediction Training Pipeline started =====")

        raw_data_path = DataIngestion().initiate_data_ingestion()

        X, y = DataTransformationRain().initiate_data_transformation(raw_data_path)

        metrics = ModelTrainerRain().initiate_model_training(X, y)

        logging.info(f"===== Rain Training Pipeline completed. Metrics: {metrics} =====")
        return metrics

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    result = run_rain_training_pipeline()
    print("Rain model trained. Metrics:", result)
