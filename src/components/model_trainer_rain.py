"""
model_trainer_rain.py
------------------------
Trains the XGBoost Classifier for Rain Prediction.
Saved model -> artifacts/rain_model.pkl
"""
import os
import sys
from dataclasses import dataclass

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerRainConfig:
    trained_model_path: str = os.path.join("artifacts", "rain_model.pkl")


class ModelTrainerRain:
    def __init__(self):
        self.config = ModelTrainerRainConfig()

    def initiate_model_training(self, X, y):
        try:
            split_idx = int(len(X) * 0.85)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            model = XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                eval_metric="logloss",
                early_stopping_rounds=30,
            )

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            preds = model.predict(X_test)

            accuracy = float(accuracy_score(y_test, preds))
            precision = float(precision_score(y_test, preds, zero_division=0))
            recall = float(recall_score(y_test, preds, zero_division=0))
            f1 = float(f1_score(y_test, preds, zero_division=0))

            logging.info(
                f"Rain model -> Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, "
                f"Recall: {recall:.3f}, F1: {f1:.3f}"
            )

            save_object(self.config.trained_model_path, model)

            return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation_rain import DataTransformationRain

    raw_path = DataIngestion().initiate_data_ingestion()
    X, y = DataTransformationRain().initiate_data_transformation(raw_path)
    metrics = ModelTrainerRain().initiate_model_training(X, y)
    print(metrics)
