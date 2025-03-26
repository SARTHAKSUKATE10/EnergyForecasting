import torch
import torch.nn as nn
import numpy as np
import joblib
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from ML_Models.utils.energy_forecast_utils import (
    CNN_LSTM_Model,
    compute_energy_distribution,
    create_full_feature_vector,
    convert_to_builtin_type
)

# Initialize Flask App
app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Paths to scalers & model
BASE_DIR = os.path.dirname(__file__)
X_SCALER_PATH = os.path.join(BASE_DIR, "ML_Models", "models", "X_scaler.pkl")
Y_SCALER_PATH = os.path.join(BASE_DIR, "ML_Models", "models", "y_scaler.pkl")
MODEL_PATH    = os.path.join(BASE_DIR, "ML_Models", "models", "cnn_lstm_model.pth")

# Load scalers
X_scaler = joblib.load(X_SCALER_PATH)
y_scaler = joblib.load(Y_SCALER_PATH)
print("✅ X_scaler and y_scaler loaded successfully!")

# Load Model
input_dim = 30  # Ensure this matches training config
model = CNN_LSTM_Model(input_dim).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
model.eval()
print("✅ Model loaded successfully!")

# Prediction Function
def predict_energy(features, season, period, temp, rainfall, humidity, population):
    full_features = create_full_feature_vector(features, temp, rainfall, humidity, population)
    full_features = np.array(full_features, dtype=np.float32).reshape(1, -1)
    
    # Scale input features
    X_scaled = X_scaler.transform(full_features).reshape(1, 1, -1)
    
    with torch.no_grad():
        y_pred_scaled = model(torch.tensor(X_scaled, dtype=torch.float32).to(device)).cpu().numpy().flatten()[0]
    
    # Inverse transform prediction
    y_pred_original = y_scaler.inverse_transform(np.array([[y_pred_scaled]]))[0][0]
    
    # Compute energy distribution
    distribution = compute_energy_distribution(
        total_energy=y_pred_original,
        season=season,
        period=period,
        festival_effect=1.0,
        temp=temp,
        rainfall=rainfall,
        humidity=humidity,
        population=population
    )
    
    return convert_to_builtin_type(distribution)

# API Route for Predictions
@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        data = request.get_json()
        features = np.array(data["features"], dtype=np.float32)
        season = data.get("season", "Summer")
        period = data.get("period", "Pre-lockdown")
        temp = float(data.get("temp", 30.0))
        rainfall = float(data.get("rainfall", 0.0))
        humidity = float(data.get("humidity", 50.0))
        population = float(data.get("population", 1000000))
        
        predictions = predict_energy(features, season, period, temp, rainfall, humidity, population)
        return jsonify(predictions)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
