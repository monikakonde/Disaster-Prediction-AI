from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
import os

# Define file paths
flood_data_path = "flood_sample_data.xlsx"
earthquake_data_path = "earthquake_sample_data.xlsx"
flood_model_path = os.path.join(os.path.dirname(__file__), "flood_model.pkl")
earthquake_model_path = os.path.join(os.path.dirname(__file__), "earthquake_model.pkl")


# 1. Train Flood Prediction Model
def train_flood_model():
    print("🔄 Training flood prediction model...")
    df = pd.read_excel(flood_data_path)

    if not all(col in df.columns for col in ['rainfall', 'river_level', 'humidity', 'temperature', 'flood']):
        raise ValueError("Missing required columns in flood dataset.")

    X = df[['rainfall', 'river_level', 'humidity', 'temperature']]
    y = df['flood']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, flood_model_path)
    print(f"✅ Flood model saved to: {flood_model_path}")


# 2. Train Earthquake Prediction Model
def train_earthquake_model():
    print("🔄 Training earthquake prediction model...")
    df = pd.read_excel(earthquake_data_path)

    if not all(col in df.columns for col in ['magnitude', 'depth', 'ground_acceleration', 'earthquake']):
        raise ValueError("Missing required columns in earthquake dataset.")

    X = df[['magnitude', 'depth', 'ground_acceleration']]
    y = df['earthquake']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, earthquake_model_path)
    print(f"✅ Earthquake model saved to: {earthquake_model_path}")


# Main execution
if __name__ == "__main__":
    train_flood_model()
    train_earthquake_model()
    print("🎉 All models trained successfully.")
