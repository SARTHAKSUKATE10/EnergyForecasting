import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load Data ---
features_df = pd.read_csv("../data/features.csv")
target_df = pd.read_csv("../data/target.csv")

# Drop the Date column if it exists (prevents string-to-float conversion errors)
if "Date" in features_df.columns:
    features_df.drop(columns=["Date"], inplace=True)

X = features_df.values.astype(np.float32)
y = target_df.values.astype(np.float32).reshape(-1, 1)

# --- Split Train & Test Sets BEFORE Scaling ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# --- Create Separate Scalers for X and y ---
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

# Fit and transform the training data
X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# Reshape for GRU: (samples, sequence_length, features)
# Here, we treat each sample as a sequence of length 1.
X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Convert to PyTorch tensors and move to device
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

# --- Define Vanilla GRU Model ---
class VanillaGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super(VanillaGRUModel, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, 
                          num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, features)
        gru_out, _ = self.gru(x)
        # Use the output from the last time step
        out = self.dropout(gru_out[:, -1, :])
        out = self.fc(out)
        return out

input_dim = X.shape[1]
gru_model = VanillaGRUModel(input_dim).to(device)

# --- Loss and Optimizer ---
criterion = nn.MSELoss()
optimizer = optim.Adam(gru_model.parameters(), lr=0.0003, weight_decay=1e-4)

# --- Training Loop ---
num_epochs = 50
batch_size = 32

dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    gru_model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = gru_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

# --- Save the Vanilla GRU Model & Scalers ---
MODEL_SAVE_PATH = "../models/vanilla_gru_model.pth"
torch.save(gru_model.state_dict(), MODEL_SAVE_PATH)
joblib.dump(X_scaler, "../models/X1_scaler.pkl")  # Save as X1_scaler.pkl for GRU
joblib.dump(y_scaler, "../models/y1_scaler.pkl")  # Save as y1_scaler.pkl for GRU
print("✅ Vanilla GRU model training completed and saved!")
