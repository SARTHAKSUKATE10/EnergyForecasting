import torch
import joblib
import json
import pandas as pd
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

        basic_features = np.array(data["features"], dtype=np.float32)
        season = data.get("season", "Summer")
        period = data.get("period", "Pre-lockdown")
        temp = float(data.get("temp", 30.0))
        rainfall = float(data.get("rainfall", 0.0))
        humidity = float(data.get("humidity", 50.0))
        population = float(data.get("population", 1000000))

        # Create full 30-dim feature vector
        full_features = create_full_feature_vector(
            basic_features, temp, rainfall, humidity, population
        )
        full_features = np.array(full_features, dtype=np.float32).reshape(1, -1)

        # Check that the feature vector is exactly 30
        if full_features.shape[1] != 30:
            return jsonify({"error": f"Expected 30 input features, got {full_features.shape[1]}"}), 400

        # Scale input
        X_scaled = X_scaler.transform(full_features)  # shape: (1, 30)
        X_scaled = X_scaled.reshape(1, 1, 30)         # CNN expects: (batch_size=1, channels=1, sequence_length=30)

        # Predict
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            y_pred_scaled = model(X_tensor).cpu().numpy().flatten()[0]

        # Inverse scale output
        y_pred_original = y_scaler.inverse_transform([[y_pred_scaled]])[0][0]

        # Compute distribution
        distribution = compute_energy_distribution(
            total_energy=y_pred_original,
            season=season,
            period=period,
            festival_effect=1.0,  # if using festival label, you can update this
            temp=temp,
            rainfall=rainfall,
            humidity=humidity,
            population=population
        )

        return jsonify(convert_to_builtin_type(distribution))

    except Exception as e:
        return jsonify({"error": str(e)}), 400


####################################
# Historical Data Endpoint
####################################
  
@app.route("/predict_monthly_energy", methods=["GET"])
def predict_monthly_energy():
    try:
        
        year = int(request.args.get("year", 2026))

        # Load monthly features JSON
        json_path = os.path.join(BASE_DIR, "ML_Models", "data", "monthly_features.json")
        with open(json_path, "r") as f:
            all_months_info = json.load(f)

        # Filter data for the selected year
        months_info = [m for m in all_months_info if m["year"] == year]

        monthly_predictions = []
        for idx, info in enumerate(months_info):
            # Basic features
            basic_features = np.array([idx + 1, info["festival_flag"]], dtype=np.float32)

            temp = info["temp"]
            rainfall = info["rainfall"]
            humidity = info["humidity"]
            population = 1000000  # Modify if you want dynamic population

            full_features = create_full_feature_vector(basic_features, temp, rainfall, humidity, population)
            full_features = np.array(full_features, dtype=np.float32).reshape(1, -1)

            X_scaled = X_scaler.transform(full_features)
            X_scaled = X_scaled.reshape(1, 1, 30)

            with torch.no_grad():
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
                y_pred_scaled = model(X_tensor).cpu().numpy().flatten()[0]

            y_pred_original = y_scaler.inverse_transform([[y_pred_scaled]])[0][0]

            monthly_predictions.append({
                "year": year,
                "month": info["month"],
                "predicted_energy": y_pred_original
            })

        return jsonify(monthly_predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 400




@app.route('/previous_data_analysis', methods=['GET'])
def previous_data_analysis():
    try:
        # Load the dataset
        df = pd.read_csv(r'C:\Users\jrsar\OneDrive\Desktop\FinalYearProject\backend\ML_Models\data\processed_data.csv')

        df['Date'] = pd.to_datetime(df['Date'])

        # Group by Year
        yearly_usage = df.groupby('Year')['Total Usage (kWh)'].sum().to_dict()

        # Group by Month
        monthly_usage = df.groupby('Month')['Total Usage (kWh)'].sum().to_dict()

        # Sectorwise (Urban & Rural)
        sector_usage = {
            "Urban Usage (kWh)": df['Urban Usage (kWh)'].sum(),
            "Rural Usage (kWh)": df['Rural Usage (kWh)'].sum()
        }

        # Urban Sectors
        urban_sectors = {
            "Urban Household": df['Urban Household (kWh)'].sum(),
            "Urban Industrial": df['Urban Industrial (kWh)'].sum(),
            "Urban Commercial": df['Urban Commercial (kWh)'].sum(),
            "Urban Others": df['Urban Others (kWh)'].sum()
        }

        # Rural Sectors
        rural_sectors = {
            "Rural Household": df['Rural Household (kWh)'].sum(),
            "Rural Industrial": df['Rural Industrial (kWh)'].sum(),
            "Rural Commercial": df['Rural Commercial (kWh)'].sum(),
            "Rural Others": df['Rural Others (kWh)'].sum()
        }

        # Group by Season
        season_usage = df.groupby('Season')['Total Usage (kWh)'].sum().to_dict()

        return jsonify({
            "yearly_usage": yearly_usage,
            "monthly_usage": monthly_usage,
            "sector_usage": sector_usage,
            "urban_sectors": urban_sectors,
            "rural_sectors": rural_sectors,
            "season_usage": season_usage
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

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