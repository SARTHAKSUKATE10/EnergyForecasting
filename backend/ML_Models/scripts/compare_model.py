import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from train_model import CNN_LSTM_Model  # Your CNN-LSTM model class

# Define a Vanilla GRU model for comparison
class VanillaGRUModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super(VanillaGRUModel, self).__init__()
        self.gru = torch.nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                                num_layers=num_layers, batch_first=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.dropout(gru_out[:, -1, :])
        out = self.fc(out)
        return out

# ------------------ Setup Paths ------------------
PROJECT_ROOT = r"C:\Users\jrsar\OneDrive\Desktop\FinalYearProject"
FEATURES_PATH = os.path.join(PROJECT_ROOT, "backend", "ML_Models", "data", "features.csv")
TARGET_PATH = os.path.join(PROJECT_ROOT, "backend", "ML_Models", "data", "target.csv")

# Scaler paths for CNN-LSTM model
X_SCALER_PATH = os.path.join(PROJECT_ROOT, "backend", "ML_Models", "models", "X_scaler.pkl")
Y_SCALER_PATH = os.path.join(PROJECT_ROOT, "backend", "ML_Models", "models", "y_scaler.pkl")

# Scaler paths for GRU model
X1_SCALER_PATH = os.path.join(PROJECT_ROOT, "backend", "ML_Models", "models", "X1_scaler.pkl")
Y1_SCALER_PATH = os.path.join(PROJECT_ROOT, "backend", "ML_Models", "models", "y1_scaler.pkl")

# Model paths
CNN_LSTM_MODEL_PATH = os.path.join(PROJECT_ROOT, "backend", "ML_Models", "models", "cnn_lstm_model.pth")
GRU_MODEL_PATH = os.path.join(PROJECT_ROOT, "backend", "ML_Models", "models", "vanilla_gru_model.pth")

# ------------------ Data Loading ------------------
features_df = pd.read_csv(FEATURES_PATH)
target_df = pd.read_csv(TARGET_PATH)

# Drop the "Date" column if present
if "Date" in features_df.columns:
    features_df.drop(columns=["Date"], inplace=True)

X = features_df.values.astype(np.float32)
y = target_df.values.astype(np.float32).reshape(-1, 1)

# ------------------ Split Data ------------------
# Use the same split as training (non-shuffled)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# ------------------ Load Scalers ------------------
# For CNN-LSTM
X_scaler = joblib.load(X_SCALER_PATH)
y_scaler = joblib.load(Y_SCALER_PATH)
# For GRU
X1_scaler = joblib.load(X1_SCALER_PATH)
y1_scaler = joblib.load(Y1_SCALER_PATH)

# ------------------ Transform Test Data ------------------
# For CNN-LSTM
X_test_scaled_cnn = X_scaler.transform(X_test)
y_test_scaled_cnn = y_scaler.transform(y_test)
X_test_scaled_cnn = X_test_scaled_cnn.reshape(X_test_scaled_cnn.shape[0], 1, X_test_scaled_cnn.shape[1])
# For GRU
X_test_scaled_gru = X1_scaler.transform(X_test)
y_test_scaled_gru = y1_scaler.transform(y_test)
X_test_scaled_gru = X_test_scaled_gru.reshape(X_test_scaled_gru.shape[0], 1, X_test_scaled_gru.shape[1])

# Convert to torch tensors
X_test_tensor_cnn = torch.tensor(X_test_scaled_cnn, dtype=torch.float32)
X_test_tensor_gru = torch.tensor(X_test_scaled_gru, dtype=torch.float32)

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_test_tensor_cnn = X_test_tensor_cnn.to(device)
X_test_tensor_gru = X_test_tensor_gru.to(device)

# ------------------ Load Models ------------------
input_dim = X.shape[1]  # number of features

# Load CNN-LSTM model
cnn_lstm_model = CNN_LSTM_Model(input_dim).to(device)
cnn_lstm_model.load_state_dict(torch.load(CNN_LSTM_MODEL_PATH, map_location=device))
cnn_lstm_model.eval()

# Load GRU model
gru_model = VanillaGRUModel(input_dim).to(device)
gru_model.load_state_dict(torch.load(GRU_MODEL_PATH, map_location=device))
gru_model.eval()

# ------------------ Make Predictions ------------------
with torch.no_grad():
    cnn_lstm_pred_scaled = cnn_lstm_model(X_test_tensor_cnn).cpu().numpy()
    gru_pred_scaled = gru_model(X_test_tensor_gru).cpu().numpy()

# Inverse transform predictions
cnn_lstm_pred = y_scaler.inverse_transform(cnn_lstm_pred_scaled)
gru_pred = y1_scaler.inverse_transform(gru_pred_scaled)

# ------------------ Evaluate Performance ------------------
mae_cnn_lstm = mean_absolute_error(y_test, cnn_lstm_pred)
mse_cnn_lstm = mean_squared_error(y_test, cnn_lstm_pred)
rmse_cnn_lstm = np.sqrt(mse_cnn_lstm)
r2_cnn_lstm = r2_score(y_test, cnn_lstm_pred)

mae_gru = mean_absolute_error(y_test, gru_pred)
mse_gru = mean_squared_error(y_test, gru_pred)
rmse_gru = np.sqrt(mse_gru)
r2_gru = r2_score(y_test, gru_pred)

print("CNN-LSTM Model Performance:")
print(f"  MAE: {mae_cnn_lstm:.4f}, RMSE: {rmse_cnn_lstm:.4f}, R²: {r2_cnn_lstm:.4f}")
print("\nVanilla GRU Model Performance:")
print(f"  MAE: {mae_gru:.4f}, RMSE: {rmse_gru:.4f}, R²: {r2_gru:.4f}")

# ------------------ Plot Predictions ------------------
x = np.arange(len(y_test))
plt.figure(figsize=(12, 6))
plt.plot(x, y_test.ravel(), label="Actual", marker="o", color="blue")
plt.plot(x, cnn_lstm_pred.ravel(), label="CNN-LSTM Predictions", marker="o", linestyle="--", color="red")
plt.plot(x, gru_pred.ravel(), label="GRU Predictions", marker="o", linestyle="--", color="green")
plt.xlabel("Sample Index")
plt.ylabel("Energy Usage (kWh)")
plt.title("Actual vs. Predicted Energy Usage (Test Data)")
plt.legend()
plt.grid(True)
plt.show()
