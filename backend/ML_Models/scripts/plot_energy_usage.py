import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import torch
import joblib

# ------------------------------------------------------------------
# 1. Setup Logging
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------------------------------------------------------
# 2. Fix Module Import Path
# ------------------------------------------------------------------
PROJECT_ROOT = r"C:\Users\jrsar\OneDrive\Desktop\FinalYearProject"
BACKEND_PATH = os.path.join(PROJECT_ROOT, "backend")
ML_MODELS_PATH = os.path.join(BACKEND_PATH, "ML_Models")

if ML_MODELS_PATH not in sys.path:
    sys.path.append(ML_MODELS_PATH)  # Ensure ML_Models is in the Python path

# Verify if the module can be imported
try:
    from utils.energy_forecast_utils import (
        CNN_LSTM_Model,
        compute_energy_distribution,
        create_full_feature_vector
    )
except ModuleNotFoundError as e:
    logging.error(f"Failed to import ML_Models module: {e}")
    sys.exit(1)

# ------------------------------------------------------------------
# 3. Load Historical Data
# ------------------------------------------------------------------
DATASET_PATH = os.path.join(ML_MODELS_PATH, "data", "sectorwise_energy_updated.csv")
X_SCALER_PATH = os.path.join(ML_MODELS_PATH, "models", "X_scaler.pkl")
Y_SCALER_PATH = os.path.join(ML_MODELS_PATH, "models", "y_scaler.pkl")
MODEL_PATH    = os.path.join(ML_MODELS_PATH, "models", "cnn_lstm_model.pth")

# Check if dataset exists
if not os.path.exists(DATASET_PATH):
    logging.error(f"Dataset not found: {DATASET_PATH}")
    sys.exit(1)

df = pd.read_csv(DATASET_PATH)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month

energy_col = "Total Usage (kWh)"
monthly_avg = df.groupby(["year", "month"])[energy_col].mean().reset_index()
pivot_table = monthly_avg.pivot(index="month", columns="year", values=energy_col)

# ------------------------------------------------------------------
# 4. Load Model & Scalers
# ------------------------------------------------------------------
if not os.path.exists(X_SCALER_PATH) or not os.path.exists(Y_SCALER_PATH) or not os.path.exists(MODEL_PATH):
    logging.error("Model or scalers not found in the specified directory.")
    sys.exit(1)

X_scaler = joblib.load(X_SCALER_PATH)
y_scaler = joblib.load(Y_SCALER_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 30
model = CNN_LSTM_Model(input_dim).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
model.eval()

# ------------------------------------------------------------------
# 5. Predict for Oct 2024 - Dec 2025
# ------------------------------------------------------------------
month_season_map = {
    1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer",
    5: "Summer", 6: "Monsoon", 7: "Monsoon", 8: "Monsoon",
    9: "Monsoon", 10: "Autumn", 11: "Autumn", 12: "Winter"
}

predicted_usage = []
months = list(range(10, 13)) + list(range(1, 13))  # Oct 2024 - Dec 2025
years = [2024] * 3 + [2025] * 12

# Default environmental values
default_temp = 30.0
default_rainfall = 0.0
default_humidity = 50.0
default_population = 4000000
default_festival = 0

for year, month in zip(years, months):
    date_str = f"{year}-{month:02d}-15"
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    timestamp = date_obj.timestamp()

    basic_features = np.array([timestamp, default_festival], dtype=np.float32)
    full_features = create_full_feature_vector(
        basic_features, default_temp, default_rainfall,
        default_humidity, default_population
    )
    full_features = np.array(full_features, dtype=np.float32).reshape(1, -1)

    # Scale input
    X_scaled = X_scaler.transform(full_features)
    X_scaled = X_scaled.reshape(1, 1, -1)

    # Predict
    with torch.no_grad():
        y_pred_scaled = model(torch.tensor(X_scaled, dtype=torch.float32).to(device)).cpu().numpy().flatten()[0]

    # Inverse transform
    y_pred_original = y_scaler.inverse_transform(np.array([[y_pred_scaled]]))[0][0]

    # Adjust predictions
    season = month_season_map.get(month, "Summer")
    distribution = compute_energy_distribution(
        total_energy=y_pred_original,
        season=season,
        period="Pre-lockdown",
        festival_effect=1.0,
        temp=default_temp,
        rainfall=default_rainfall,
        humidity=default_humidity,
        population=default_population
    )

    total_energy_pred = distribution["Total Energy"]

    if season == "Monsoon":
        total_energy_pred = max(9200, min(total_energy_pred, 9800))
    elif season == "Autumn":
        total_energy_pred = max(9500, min(total_energy_pred, 10000))
    elif season == "Summer":
        total_energy_pred = max(9800, min(total_energy_pred, 11000))
    else:
        total_energy_pred = max(9300, min(total_energy_pred, 9700))

    predicted_usage.append(total_energy_pred)

# ------------------------------------------------------------------
# 6. Plot Actual and Predicted Data
# ------------------------------------------------------------------
plt.figure(figsize=(12, 6))

# Plot historical data
for yr in pivot_table.columns:
    plt.plot(pivot_table.index, pivot_table[yr], marker='o', linestyle='-', label=str(yr))

# Plot predictions
plt.plot(months, predicted_usage, marker='o', linestyle='--', color='red', linewidth=2, label="Predicted 2024-2025")

plt.xlabel("Month")
plt.ylabel("Average Energy Usage (kWh)")
plt.title("Historical vs. Predicted Energy Usage (Oct 2024 - Dec 2025)")
plt.xticks(months, ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.grid(True)
plt.legend()
plt.show()
