import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import joblib

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ML_Models.utils.energy_forecast_utils import (
    CNN_LSTM_Model,
    compute_energy_distribution,
    create_full_feature_vector
)

# ------------------------------------------------------------------
# 1. Load Historical Data
# ------------------------------------------------------------------
PROJECT_ROOT = r"C:\Users\jrsar\OneDrive\Desktop\FinalYearProject"
DATASET_PATH = os.path.join(PROJECT_ROOT, "backend", "ML_Models", "data", "sectorwise_energy_updated.csv")
X_SCALER_PATH = os.path.join(PROJECT_ROOT, "backend", "ML_Models", "models", "X_scaler.pkl")
Y_SCALER_PATH = os.path.join(PROJECT_ROOT, "backend", "ML_Models", "models", "y_scaler.pkl")
MODEL_PATH    = os.path.join(PROJECT_ROOT, "backend", "ML_Models", "models", "cnn_lstm_model.pth")

df = pd.read_csv(DATASET_PATH)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month

energy_col = "Total Usage (kWh)"
monthly_avg = df.groupby(["year", "month"])[energy_col].mean().reset_index()

# ------------------------------------------------------------------
# 2. Load Model & Scalers
# ------------------------------------------------------------------
X_scaler = joblib.load(X_SCALER_PATH)
y_scaler = joblib.load(Y_SCALER_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 30
model = CNN_LSTM_Model(input_dim).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
model.eval()

# ------------------------------------------------------------------
# 3. Predict Energy Usage for Oct 2024 - Dec 2025
# ------------------------------------------------------------------
predicted_usage = []
date_range = pd.date_range(start="2024-10-01", periods=15, freq="M")

for date in date_range:
    timestamp = date.timestamp()
    basic_features = np.array([timestamp, 0], dtype=np.float32)
    full_features = create_full_feature_vector(basic_features, 30.0, 0.0, 50.0, 4000000)
    full_features = np.array(full_features, dtype=np.float32).reshape(1, -1)
    
    X_scaled = X_scaler.transform(full_features)
    X_scaled = X_scaled.reshape(1, 1, -1)

    with torch.no_grad():
        y_pred_scaled = model(torch.tensor(X_scaled, dtype=torch.float32).to(device)).cpu().numpy().flatten()[0]
    
    y_pred_original = y_scaler.inverse_transform(np.array([[y_pred_scaled]]))[0][0]
    predicted_usage.append((date.year, date.month, y_pred_original))

# Convert predictions into DataFrame
predicted_df = pd.DataFrame(predicted_usage, columns=["year", "month", "predicted_usage"])

# ------------------------------------------------------------------
# 4. Plot Historical and Predicted Data
# ------------------------------------------------------------------
plt.figure(figsize=(12, 6))

# Plot historical data
plt.plot(
    monthly_avg["year"] + (monthly_avg["month"] - 1) / 12,
    monthly_avg[energy_col],
    marker="o", linestyle="-", color="blue", label="Historical Usage"
)

# Plot predicted data
plt.plot(
    predicted_df["year"] + (predicted_df["month"] - 1) / 12,
    predicted_df["predicted_usage"],
    marker="o", linestyle="--", color="red", linewidth=2, label="Predicted Usage (2024-2025)"
)

plt.xlabel("Year")
plt.ylabel("Average Energy Usage (kWh)")
plt.title("Historical vs. Predicted Energy Usage (2024-2025)")
plt.grid(True)
plt.legend()
plt.show()