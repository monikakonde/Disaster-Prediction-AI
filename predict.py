import joblib
import numpy as np
import os

# Load models from models/ directory
flood_model = joblib.load(os.path.join("flood_model.pkl"))
earthquake_model = joblib.load(os.path.join("earthquake_model.pkl"))


def predict_flood(data):
    features = np.array([[data['rainfall'], data['river_level'], data['humidity'], data['temperature']]])
    prediction = flood_model.predict(features)[0]
    return "🌊 Flood Risk: YES" if prediction == 1 else "✅ Flood Risk: NO"


def predict_earthquake(data):
    features = np.array([[data['magnitude'], data['depth'], data['ground_acceleration']]])
    prediction = earthquake_model.predict(features)[0]
    return "⚠️ Earthquake Risk: YES" if prediction == 1 else "✅ Earthquake Risk: NO"
