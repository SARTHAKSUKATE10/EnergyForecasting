import torch
import torch.nn as nn
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# --- Load Data ---
features_df = pd.read_csv("../data/features.csv")
target_df = pd.read_csv("../data/target.csv")

if "Date" in features_df.columns:
    features_df = features_df.drop(columns=["Date"])

X = features_df.values.astype(np.float32)
y = target_df.values.astype(np.float32).reshape(-1, 1)

# --- Load Scalers ---
X_scaler = joblib.load("../models/X_scaler.pkl")  
y_scaler = joblib.load("../models/y_scaler.pkl")

# --- Scale Data ---
X_scaled = X_scaler.transform(X)
y_scaled = y_scaler.transform(y)

# --- Split Data (Same as in training) ---
split_index = int(len(X) * 0.8)
X_test_scaled = X_scaled[split_index:]
y_test_scaled = y_scaled[split_index:]

# Reshape for models
X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Convert to PyTorch tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

# --- Define Models (Matching Saved Versions) ---
class LSTM_Model(nn.Module):
    def __init__(self, input_dim):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=32, num_layers=2, batch_first=True)  # Matched previous model
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        return self.fc(x)

class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_dim):
        super(CNN_LSTM_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # Swap dimensions for LSTM
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class GRU_Model(nn.Module):
    def __init__(self, input_dim):
        super(GRU_Model, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=32, num_layers=1, batch_first=True)  # Matched previous GRU
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.dropout(gru_out[:, -1, :])
        return self.fc(out)

# --- Load Models ---
input_dim = X.shape[1]

lstm_model = LSTM_Model(input_dim).to(device)
lstm_model.load_state_dict(torch.load("../models/lstm_model.pth"))
lstm_model.eval()

cnn_lstm_model = CNN_LSTM_Model(input_dim).to(device)
cnn_lstm_model.load_state_dict(torch.load("../models/cnn_lstm_model.pth"))
cnn_lstm_model.eval()

gru_model = GRU_Model(input_dim).to(device)
gru_model.load_state_dict(torch.load("../models/gru_model.pth"))
gru_model.eval()

# --- Evaluation Function ---
def evaluate_model(model, X_test, y_test, y_scaler, model_name):
    with torch.no_grad():
        y_pred_scaled = model(X_test).cpu().numpy()
        y_test_scaled = y_test.cpu().numpy()

    # Inverse transform to get original scale
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_test_scaled)

    # Calculate Metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n📊 {model_name} Performance:")
    print(f"🔹 MSE  : {mse:.4f}")
    print(f"🔹 RMSE : {rmse:.4f}")
    print(f"🔹 R² Score: {r2:.4f}")

    return mse, rmse, r2

# --- Compare Models ---
lstm_results = evaluate_model(lstm_model, X_test_tensor, y_test_tensor, y_scaler, "LSTM Model")
cnn_lstm_results = evaluate_model(cnn_lstm_model, X_test_tensor, y_test_tensor, y_scaler, "CNN-LSTM Model")
gru_results = evaluate_model(gru_model, X_test_tensor, y_test_tensor, y_scaler, "GRU Model")

# --- Tabular Comparison ---
results_df = pd.DataFrame({
    "Model": ["LSTM", "CNN-LSTM", "GRU"],
    "MSE": [lstm_results[0], cnn_lstm_results[0], gru_results[0]],
    "RMSE": [lstm_results[1], cnn_lstm_results[1], gru_results[1]],
    "R² Score": [lstm_results[2], cnn_lstm_results[2], gru_results[2]]
})

print("\n📌 Model Comparison:\n")
print(results_df)
