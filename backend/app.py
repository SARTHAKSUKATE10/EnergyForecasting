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

# Frontend Path
FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend")
app = Flask(
    __name__,
    template_folder=os.path.join(FRONTEND_PATH, "templates"),
    static_folder=os.path.join(FRONTEND_PATH, "static")
)
CORS(app)

# Model and Scaler paths
BASE_DIR = os.path.dirname(__file__)
X_SCALER_PATH = os.path.join(BASE_DIR, "ML_Models", "models", "X_scaler.pkl")
Y_SCALER_PATH = os.path.join(BASE_DIR, "ML_Models", "models", "y_scaler.pkl")
MODEL_PATH    = os.path.join(BASE_DIR, "ML_Models", "models", "cnn_lstm_model.pth")

# Load scalers
X_scaler = joblib.load(X_SCALER_PATH)
y_scaler = joblib.load(Y_SCALER_PATH)
print("✅ X_scaler and y_scaler loaded successfully!")

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 30
model = CNN_LSTM_Model(input_dim).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
model.eval()
print("✅ Model loaded successfully!")

####################################
# Define Routes
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
# Prediction Route
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

        full_features = create_full_feature_vector(
            basic_features, temp, rainfall, humidity, population
        )
        full_features = np.array(full_features, dtype=np.float32).reshape(1, -1)

        if full_features.shape[1] != 30:
            return jsonify({"error": f"Expected 30 input features, got {full_features.shape[1]}"}), 400

        X_scaled = X_scaler.transform(full_features)
        X_scaled = X_scaled.reshape(1, 1, 30)

        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            y_pred_scaled = model(X_tensor).cpu().numpy().flatten()[0]

        y_pred_original = y_scaler.inverse_transform([[y_pred_scaled]])[0][0]

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

        return jsonify(convert_to_builtin_type(distribution))

    except Exception as e:
        return jsonify({"error": str(e)}), 400

####################################
# Corrected Future Prediction
####################################

@app.route("/predict_monthly_energy", methods=["GET"])
def predict_monthly_energy():
    try:
        year = int(request.args.get("year", 2026))

        months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]

        season_mapping = {
            1: "Winter", 2: "Winter", 3: "Summer",
            4: "Summer", 5: "Summer", 6: "Monsoon",
            7: "Monsoon", 8: "Monsoon", 9: "Autumn",
            10: "Autumn", 11: "Winter", 12: "Winter"
        }

        # Monthly Averages
        monthly_avg = {
            1: {"temp": 20.70, "rainfall": 0.47, "humidity": 65.16, "population": 3130000},
            2: {"temp": 22.70, "rainfall": 0.018, "humidity": 53.93, "population": 3130000},
            3: {"temp": 25.71, "rainfall": 0.168, "humidity": 50.24, "population": 3130000},
            4: {"temp": 28.54, "rainfall": 0.524, "humidity": 52.13, "population": 3130000},
            5: {"temp": 28.49, "rainfall": 1.43, "humidity": 62.93, "population": 3130000},
            6: {"temp": 26.02, "rainfall": 6.02, "humidity": 80.05, "population": 3130000},
            7: {"temp": 24.15, "rainfall": 8.52, "humidity": 89.20, "population": 3130000},
            8: {"temp": 23.85, "rainfall": 4.75, "humidity": 88.55, "population": 3130000},
            9: {"temp": 24.12, "rainfall": 6.61, "humidity": 86.74, "population": 3130000},
            10: {"temp": 24.43, "rainfall": 5.60, "humidity": 77.01, "population": 3130000},
            11: {"temp": 22.98, "rainfall": 1.09, "humidity": 70.53, "population": 3130000},
            12: {"temp": 21.21, "rainfall": 0.57, "humidity": 69.60, "population": 3130000},
        }

        monthly_predictions = []

        for idx, month_name in enumerate(months):
            month_num = idx + 1
            averages = monthly_avg[month_num]

            temp = averages["temp"]
            rainfall = averages["rainfall"]
            humidity = averages["humidity"]
            population = averages["population"]

            festival_flag = 1 if month_num in [10, 11, 12] else 0
            season = season_mapping.get(month_num, "Summer")

            # Build 30-feature vector
            feature_vector = np.zeros(30, dtype=np.float32)
            feature_vector[0] = month_num
            feature_vector[1] = festival_flag
            feature_vector[4] = temp
            feature_vector[5] = rainfall
            feature_vector[6] = humidity
            feature_vector[7] = population

            # Season encoding
            if season == "Winter":
                feature_vector[26] = 1
            elif season == "Summer":
                feature_vector[25] = 1
            elif season == "Monsoon":
                feature_vector[24] = 1
            elif season == "Autumn":
                feature_vector[23] = 1

            # Period encoding
            feature_vector[28] = 1  # Post-lockdown always

            feature_vector = feature_vector.reshape(1, -1)

            # Scale input
            X_scaled = X_scaler.transform(feature_vector)
            X_scaled = X_scaled.reshape(1, 1, 30)

            # Predict
            with torch.no_grad():
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
                y_pred_scaled = model(X_tensor).cpu().numpy().flatten()[0]

            y_pred_original = y_scaler.inverse_transform([[y_pred_scaled]])[0][0]

            # 🌟 Adjust prediction using compute_energy_distribution
            adjusted_distribution = compute_energy_distribution(
                total_energy=y_pred_original,
                season=season,
                period="Post-lockdown",
                temp=temp,
                rainfall=rainfall,
                humidity=humidity,
                population=population
            )

            adjusted_energy = adjusted_distribution["Total Energy"]

            monthly_predictions.append({
                "year": year,
                "month": month_name,
                "predicted_energy": adjusted_energy
            })

        return jsonify(monthly_predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 400
####################################
# Previous Data Analysis (untouched)
####################################

@app.route('/previous_data_analysis', methods=['GET'])
def previous_data_analysis():
    try:
        df = pd.read_csv(r'C:\Users\jrsar\OneDrive\Desktop\FinalYearProject\backend\ML_Models\data\processed_data.csv')

        df['Date'] = pd.to_datetime(df['Date'])

        yearly_usage = df.groupby('Year')['Total Usage (kWh)'].sum().to_dict()
        monthly_usage = df.groupby('Month')['Total Usage (kWh)'].sum().to_dict()

        sector_usage = {
            "Urban Usage (kWh)": df['Urban Usage (kWh)'].sum(),
            "Rural Usage (kWh)": df['Rural Usage (kWh)'].sum()
        }

        urban_sectors = {
            "Urban Household": df['Urban Household (kWh)'].sum(),
            "Urban Industrial": df['Urban Industrial (kWh)'].sum(),
            "Urban Commercial": df['Urban Commercial (kWh)'].sum(),
            "Urban Others": df['Urban Others (kWh)'].sum()
        }

        rural_sectors = {
            "Rural Household": df['Rural Household (kWh)'].sum(),
            "Rural Industrial": df['Rural Industrial (kWh)'].sum(),
            "Rural Commercial": df['Rural Commercial (kWh)'].sum(),
            "Rural Others": df['Rural Others (kWh)'].sum()
        }

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
# Run
####################################

if __name__ == "__main__":
    app.run(debug=True)
