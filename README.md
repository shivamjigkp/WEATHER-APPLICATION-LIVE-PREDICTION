# Weather Intelligence App

A weather platform that goes beyond simple "current weather" lookup by adding
three Machine Learning powered features on top of the existing React weather
app:

| Feature | Type | File naming pattern |
|---|---|---|
| Temperature Forecasting (next 24hr) | Regression (XGBoost) | `*_temperature.py` |
| Rain Prediction (rain tomorrow: yes/no) | Classification (XGBoost/RandomForest) | `*_rain.py` |
| Weather Anomaly Detection | Statistical + Isolation Forest | `*_anomaly.py` |

## Project Structure

```
Weather-Intelligence-App/
├── data/
│   ├── raw/                        <- raw generated/downloaded csv sits here
│   └── processed/                  <- cleaned + feature-engineered data
├── src/
│   ├── logger.py                   <- logging setup (used everywhere)
│   ├── exception.py                <- custom exception class
│   ├── utils.py                    <- save_object/load_object helpers
│   ├── components/
│   │   ├── data_ingestion.py               <- generates/loads raw weather data
│   │   ├── data_transformation_temperature.py  <- cleaning + features for temp model
│   │   ├── data_transformation_rain.py         <- cleaning + features for rain model
│   │   ├── model_trainer_temperature.py        <- trains XGBoost Regressor
│   │   └── model_trainer_rain.py               <- trains XGBoost Classifier
│   └── pipeline/
│       ├── train_pipeline_temperature.py       <- runs full temperature training
│       ├── train_pipeline_rain.py              <- runs full rain training
│       ├── predict_pipeline_temperature.py     <- loads model, predicts temp
│       ├── predict_pipeline_rain.py            <- loads model, predicts rain
│       └── anomaly_detection.py                <- detects unusual weather
├── artifacts/                      <- trained models (.pkl) saved here after training
├── notebook/                       <- EDA notebooks
├── flask_app/
│   └── app.py                      <- API endpoints, called by the React frontend
├── requirements.txt
├── setup.py
└── README.md
```

## How to run (once you clone this)

```bash
pip install -r requirements.txt

# Step 1: Train the temperature forecasting model
python src/pipeline/train_pipeline_temperature.py

# Step 2: Train the rain prediction model
python src/pipeline/train_pipeline_rain.py

# Step 3: Start the Flask API (serves both models + anomaly check)
python flask_app/app.py
```

The Flask API exposes:
- `POST /predict-temperature` -> next 24hr temperature forecast
- `POST /predict-rain` -> rain tomorrow yes/no + probability
- `POST /check-anomaly` -> whether current weather reading is unusual

The existing React app (`Weather_Application_React-main`) calls these endpoints
alongside its existing OpenWeatherMap current-weather call.

## Note on data

`src/components/data_ingestion.py` currently **generates a realistic synthetic
hourly weather dataset** (2 years, with seasonal + daily patterns, temperature,
humidity, pressure, wind speed, rain flag) so the entire pipeline is runnable
end-to-end immediately without needing an API key.

When you're ready, swap this out for a real historical dataset (Kaggle "Rain
in Australia", or OpenWeatherMap's historical API) — the rest of the pipeline
(transformation, training, prediction) will keep working as long as the
column names match.
