"""
flask_app/app.py
--------------------
API layer. The existing React weather app (Weather_Application_React-main)
already calls OpenWeatherMap directly for "current weather". This Flask app
adds the ML-powered endpoints on top of that:

    POST /predict-temperature   -> temperature forecast (next 24hr)
    POST /predict-rain          -> rain prediction (next 24hr)
    POST /check-anomaly         -> is current weather unusual?

Run with: python flask_app/app.py   (from the project root)
"""
import sys

sys.path.append(".")  # allow `from src...` imports when run from project root

from flask import Flask, jsonify, request
from flask_cors import CORS

from src.pipeline.anomaly_detection import AnomalyDetector
from src.pipeline.predict_pipeline_rain import PredictPipelineRain
from src.pipeline.predict_pipeline_temperature import PredictPipelineTemperature

app = Flask(__name__)
CORS(app)  # allow the React app (different port) to call this API


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Weather Intelligence API is running"})


@app.route("/predict-temperature", methods=["POST"])
def predict_temperature():
    """Body: { "lat": 28.6139, "lon": 77.2090 }
    Fetches REAL recent history for this location (Open-Meteo) and predicts."""
    data = request.get_json()
    try:
        prediction = PredictPipelineTemperature().predict_from_location(
            lat=data["lat"], lon=data["lon"]
        )
        return jsonify({"forecast_temperature_next_24hr": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict-rain", methods=["POST"])
def predict_rain():
    """Body: { "lat": 28.6139, "lon": 77.2090 }"""
    data = request.get_json()
    try:
        will_rain, probability = PredictPipelineRain().predict_from_location(
            lat=data["lat"], lon=data["lon"]
        )
        return jsonify({"will_rain_next_24hr": bool(will_rain), "probability": round(probability, 3)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/check-anomaly", methods=["POST"])
def check_anomaly():
    data = request.get_json()
    try:
        result = AnomalyDetector().check(
            current_temperature=data["temperature"],
            current_month=data["month"],
        )
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
