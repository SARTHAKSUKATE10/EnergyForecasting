import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split  # ✅ Add this
from train_model import CNN_LSTM_Model
from lstm_model import LSTM_Model
from gru_model import WeakGRUModel

# --- Check Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# --- Load Dataset ---
features_df = pd.read_csv("../data/features.csv")
target_df = pd.read_csv("../data/target.csv")

if "Date" in features_df.columns:
    features_df.drop(columns=["Date"], inplace=True)

X = features_df.values.astype(np.float32)
y = target_df.values.astype(np.float32).reshape(-1, 1)

# --- Split Test Set ---
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# --- Helper function to load and predict ---
def predict_with_model(model_class, model_path, x_scaler_path, y_scaler_path, input_shape, name):
    # Load scalers
    X_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)

    X_test_scaled = X_scaler.transform(X_test).reshape(X_test.shape[0], 1, input_shape)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    # Load model
    model = model_class(input_shape).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).cpu().numpy()

    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    return y_pred.flatten()

# --- Predict for all models ---
input_dim = X.shape[1]

y_pred_cnn_lstm = predict_with_model(CNN_LSTM_Model, "../models/cnn_lstm_model.pth", "../models/X_scaler.pkl", "../models/y_scaler.pkl", input_dim, "CNN-LSTM")
y_pred_lstm      = predict_with_model(LSTM_Model, "../models/lstm_model.pth", "../models/X2_scaler.pkl", "../models/y2_scaler.pkl", input_dim, "LSTM")
y_pred_gru       = predict_with_model(WeakGRUModel, "../models/gru_model.pth", "../models/X1_scaler.pkl", "../models/y1_scaler.pkl", input_dim, "GRU")

# --- Plot ---
plt.figure(figsize=(14, 6))
plt.plot(y_test, label="Actual", color="black", linewidth=2)
plt.plot(y_pred_cnn_lstm, label="CNN-LSTM", linestyle="--", color="red")
plt.plot(y_pred_lstm, label="LSTM", linestyle="--", color="green")
plt.plot(y_pred_gru, label="GRU", linestyle="--", color="blue")
plt.title("Energy Usage Prediction - Model Comparison")
plt.xlabel("Sample Index")
plt.ylabel("Energy Usage (kWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
