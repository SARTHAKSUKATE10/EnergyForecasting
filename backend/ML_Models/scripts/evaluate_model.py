import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from train_model import CNN_LSTM_Model  
import joblib
import os

# Checking if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Loading the Scalers ---
X_scaler = joblib.load("../models/X_scaler.pkl")
y_scaler = joblib.load("../models/y_scaler.pkl")
print("Scalers loaded successfully!")

# --- Loading Data ---
features_df = pd.read_csv("../data/features.csv")
target_df = pd.read_csv("../data/target.csv")


if "Date" in features_df.columns:
    features_df = features_df.drop(columns=["Date"])

X = features_df.values.astype(np.float32)
y = target_df.values.astype(np.float32).reshape(-1, 1)

# --- Spliting Train & Test Sets BEFORE Scaling ---
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# --- Transforming Using the Loaded X_scaler and y_scaler ---
X_test_scaled = X_scaler.transform(X_test)
y_test_scaled = y_scaler.transform(y_test)

# Reshaping for CNN-LSTM: (samples, 1, number of features)
X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

# --- Initializing & Load the Trained Model ---
input_dim = X.shape[1]
model = CNN_LSTM_Model(input_dim).to(device)
model.load_state_dict(torch.load("../models/cnn_lstm_model.pth", map_location=device))
model.eval()
print("Model loaded successfully!")

# --- Make Predictions ---
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor).squeeze()


y_pred_scaled = y_pred_tensor.cpu().numpy().reshape(-1, 1)

# Inverse transforming predictions and actual values using y_scaler
y_pred_original = y_scaler.inverse_transform(y_pred_scaled)
y_test_original = y_scaler.inverse_transform(y_test_scaled)


mae = mean_absolute_error(y_test_original, y_pred_original)
mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred_original)

print("\n Model Performance (Original Scale):")
print(f" MAE: {mae:.4f}")
print(f" RMSE: {rmse:.4f}")
print(f" R² Score: {r2:.4f}")

# --- Ploting Actual vs Predicted ---
plt.figure(figsize=(12, 6))
plt.scatter(range(len(y_test_original)), y_test_original, label="Actual", alpha=0.7, color='blue')
plt.scatter(range(len(y_pred_original)), y_pred_original, label="Predicted", alpha=0.7, color='orange')
plt.xlabel("Samples")
plt.ylabel("Energy Usage (kWh)")
plt.title("Actual vs. Predicted Energy Usage (Original Scale)")
plt.legend()
plt.grid(True)
plt.show()
