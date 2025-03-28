import torch
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import random

from ML_Models.utils.energy_forecast_utils import (
    CNN_LSTM_Model,
    compute_energy_distribution,
    create_full_feature_vector,
    convert_to_builtin_type
)

# Set the path for the frontend folder (assumes backend and frontend are sibling folders)
FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend")
app = Flask(
    __name__,
    template_folder=os.path.join(FRONTEND_PATH, "templates"),
    static_folder=os.path.join(FRONTEND_PATH, "static")
)
CORS(app)

####################################
# Paths to scalers & model
####################################
BASE_DIR = os.path.dirname(__file__)
X_SCALER_PATH = os.path.join(BASE_DIR, "ML_Models", "models", "X_scaler.pkl")
Y_SCALER_PATH = os.path.join(BASE_DIR, "ML_Models", "models", "y_scaler.pkl")
MODEL_PATH    = os.path.join(BASE_DIR, "ML_Models", "models", "cnn_lstm_model.pth")

# Load scalers
X_scaler = joblib.load(X_SCALER_PATH)
y_scaler = joblib.load(Y_SCALER_PATH)
print("✅ X_scaler and y_scaler loaded successfully!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################################
# Load Model
####################################
input_dim = 30  # Must match training code
model = CNN_LSTM_Model(input_dim).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
model.eval()
print("✅ Model loaded successfully!")

####################################
# Define Routes for All Pages
####################################
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

####################################
# Prediction Endpoint
####################################
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        # Expected input example:
        # {
        #   "features": [20250302, 1],
        #   "season": "Summer",
        #   "period": "Post-lockdown",
        #   "temp": 34.0,
        #   "rainfall": 5.0,
        #   "humidity": 69.0,
        #   "population": 4000000
        # }
        basic_features = np.array(data["features"], dtype=np.float32)
        season = data.get("season", "Summer")
        period = data.get("period", "Pre-lockdown")
        temp = float(data.get("temp", 30.0))
        rainfall = float(data.get("rainfall", 0.0))
        humidity = float(data.get("humidity", 50.0))
        population = float(data.get("population", 1000000))

        full_features = create_full_feature_vector(basic_features, temp, rainfall, humidity, population)
        full_features = np.array(full_features, dtype=np.float32).reshape(1, -1)

        # Scale features with X_scaler
        X_scaled = X_scaler.transform(full_features)
        X_scaled = X_scaled.reshape(1, 1, -1)

        # Predict in scaled target space
        with torch.no_grad():
            y_pred_scaled = model(torch.tensor(X_scaled, dtype=torch.float32).to(device)).cpu().numpy().flatten()[0]

        # Inverse transform with y_scaler to get prediction in original scale
        y_pred_original = y_scaler.inverse_transform(np.array([[y_pred_scaled]]))[0][0]

        # Compute full distribution using environmental adjustments
        distribution = compute_energy_distribution(
            total_energy=y_pred_original,
            season=season,
            period=period,
            festival_effect=1.0,  # Adjust if needed
            temp=temp,
            rainfall=rainfall,
            humidity=humidity,
            population=population
        )

        updated_distribution = convert_to_builtin_type(distribution)
        return jsonify(updated_distribution)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

####################################
# Historical Data Endpoint
####################################
@app.route("/get_historical_data")
def get_historical_data():
    try:
        # Generate sample historical data
        years = list(range(2018, 2025))
        total_consumption = []
        
        # Generate yearly consumption data (example: increasing trend)
        base_consumption = 1000.0  # Base consumption in MW
        growth_rate = 0.05  # 5% growth per year
        
        for year in years:
            consumption = base_consumption * (1 + growth_rate) ** (year - 2018)
            total_consumption.append(float(consumption))

        # Generate monthly consumption data for the latest year
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        latest_year_consumption = total_consumption[-1]
        monthly_consumption = []
        
        # Generate monthly data with seasonal variations
        for month in range(12):
            # Example seasonal variation: higher in summer, lower in winter
            if month in [5, 6, 7]:  # Summer months
                multiplier = 1.2
            elif month in [11, 0, 1]:  # Winter months
                multiplier = 0.8
            else:
                multiplier = 1.0
            
            monthly_consumption.append(float(latest_year_consumption * multiplier / 12))

        # Generate sector distribution data
        sector_distribution = {
            "household": latest_year_consumption * 0.30,
            "industrial": latest_year_consumption * 0.40,
            "commercial": latest_year_consumption * 0.20,
            "others": latest_year_consumption * 0.10
        }

        # Generate renewable vs non-renewable data
        renewable_data = {
            "years": [str(y) for y in years],
            "renewable": [float(consumption * 0.30) for consumption in total_consumption],
            "non_renewable": [float(consumption * 0.70) for consumption in total_consumption]
        }

        # Generate peak demand hours data
        peak_demand = {
            "values": [2000, 2200, 3500, 4500, 4000, 3000]  # Example peak demand values
        }

        # Generate temperature vs energy consumption data
        temp_energy = [
            {"temp": 15, "energy": 2500},
            {"temp": 20, "energy": 2800},
            {"temp": 25, "energy": 3200},
            {"temp": 30, "energy": 3800},
            {"temp": 35, "energy": 4500},
            {"temp": 40, "energy": 5200}
        ]

        response = {
            "years": [str(y) for y in years],
            "total_consumption": total_consumption,
            "months": months,
            "monthly_consumption": monthly_consumption,
            "sector_distribution": sector_distribution,
            "renewable_data": renewable_data,
            "peak_demand": peak_demand,
            "temp_energy": temp_energy
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

####################################
# Additional Feature: Logging Requests
####################################
import logging

logging.basicConfig(level=logging.INFO)

@app.before_request
def log_request_info():
    logging.info(f"Request Headers: {request.headers}")
    logging.info(f"Request Body: {request.get_data()}")

####################################
# Run the App
####################################
if __name__ == "__main__":
    app.run(debug=True)