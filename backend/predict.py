import torch
import torch.nn as nn
import numpy as np
import joblib
import os
from flask import Flask, request, jsonify
from ML_Models.utils.energy_forecast_utils import compute_energy_distribution

# Initialize Flask App
app = Flask(__name__)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# --- Load the Scaler ---
scaler_path = "ML_Models/models/scaler.pkl"
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print("✅ Scaler loaded successfully!")
else:
    raise FileNotFoundError("🚨 Scaler file not found! Please run train_model.py first.")

# --- Define the CNN-LSTM Model Architecture ---
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_dim):
        super(CNN_LSTM_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # Swap dimensions for LSTM input
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# --- Instantiate the Model & Load Weights ---
input_dim = 10  # Update this value if your feature count changes
model = CNN_LSTM_Model(input_dim=input_dim).to(device)
model_weights_path = "ML_Models/models/cnn_lstm_model.pth"
if os.path.exists(model_weights_path):
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()
    print("✅ Model loaded successfully!")
else:
    raise FileNotFoundError("🚨 Model weights not found! Please run train_model.py first.")

# --- Prediction Function ---
def predict_energy(input_data, season, period, festival_effect=1.0):
    """
    Predict total energy consumption using the CNN-LSTM model,
    then compute urban, rural, and sector-wise breakdown.
    """
    # Convert input data into a NumPy array and scale it
    input_array = np.array(input_data, dtype=np.float32).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    
    # Reshape for CNN-LSTM: (batch_size, time_steps, features)
    input_scaled = input_scaled.reshape(1, 1, input_scaled.shape[1])
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
    
    # Make prediction
    with torch.no_grad():
        predicted_total_energy = model(input_tensor).cpu().numpy().flatten()[0]
    
    # Compute the energy breakdown using the utility function
    energy_breakdown = compute_energy_distribution(predicted_total_energy, season, period, festival_effect)
    return energy_breakdown

# ... (your existing code above)

# --- Test Route for Debugging ---
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Server is running. Use POST /predict to get predictions."})

# --- API Route for Predictions ---
@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "No input features provided!"}), 400

    # Extract values from the JSON request
    input_features = data.get("features", [])
    season = data.get("season", "Winter")
    period = data.get("period", "Post-lockdown")
    festival_effect = data.get("festival_effect", 1.0)
    
    predictions = predict_energy(input_features, season, period, festival_effect)
    return jsonify(predictions)

# --- Print URL Map for Debugging ---
print("Registered Routes:")
print(app.url_map)

# --- Run Flask App ---
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
