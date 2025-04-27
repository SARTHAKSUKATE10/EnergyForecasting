import torch
import torch.nn as nn
import numpy as np

####################################
# CNN-LSTM Model (must match train_model.py)
####################################
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
        x = x.permute(0, 2, 1)  
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

####################################
# compute_energy_distribution (Improved)
####################################
def compute_energy_distribution(total_energy, season, period, festival_effect=1.0,
                                temp=30.0, rainfall=0.0, humidity=50.0, population=1000000):
    """
    Returns a dict with "Total Energy", "Urban Usage", "Rural Usage",
    "Urban Distribution", and "Rural Distribution".
    """

    # Basic validation
    if total_energy < 0:
        raise ValueError("Total energy must be non-negative.")
    if rainfall < 0:
        raise ValueError("Rainfall cannot be negative.")
    if population <= 0:
        raise ValueError("Population must be positive.")

    # Base ratios
    base_urban_ratio = 0.6059
    base_rural_ratio = 0.3941

    urban_sector_ratios = {
        "Urban Household": 0.3,
        "Urban Industrial": 0.39,
        "Urban Commercial": 0.21,
        "Urban Others": 0.1,
    }
    rural_sector_ratios = {
        "Rural Household": 0.4726,
        "Rural Industrial": 0.1890,
        "Rural Commercial": 0.1890,
        "Rural Others": 0.0945,
    }

    # Multipliers
    season_multipliers = {
        "Winter": 0.895,
        "Summer": 1.209,
        "Monsoon": 0.862,
        "Autumn": 1.147,
        "Post-Monsoon":0.738
    }
    period_multipliers = {
        "Pre-lockdown": 0.72,
        "Lockdown": 0.942,
        "Post-lockdown": 1.15,
    }
    festival_multipliers = {
        "Normal Day": 1.00,
        "Festival": 1.10,
    }

    season_factor = season_multipliers.get(season, 1.0)
    period_factor = period_multipliers.get(period, 1.0)
    festival_factor = festival_multipliers.get(festival_effect, 1.0)  # Corrected

    # Weather & Population impact (Adjusted Formula)
    temp_factor = 1.0 + (temp - 30.0) / 100.0
    rainfall_factor = max(1.0 - (rainfall / 200.0), 0.5)  
    humidity_factor = 1.0 + (humidity - 50.0) / 200.0
    population_factor = 1.0 + (population - 1000000) / 10000000.0  

    # Combined Effect
    weather_population_factor = (temp_factor + rainfall_factor + humidity_factor + population_factor) / 4

    # Adjust total energy
    adjusted_total_energy = total_energy * season_factor * period_factor * festival_factor * weather_population_factor

    # Scaling the total energy to avoid unrealistic values
    adjusted_total_energy = max(adjusted_total_energy, total_energy * 0.7)  # Lower bound at 70% of predicted
    adjusted_total_energy = min(adjusted_total_energy, total_energy * 1.3)  # Upper bound at 130% of predicted

    # Compute distributions
    urban_usage = adjusted_total_energy * base_urban_ratio
    rural_usage = adjusted_total_energy * base_rural_ratio

    urban_distribution = {k: urban_usage * v for k, v in urban_sector_ratios.items()}
    rural_distribution = {k: rural_usage * v for k, v in rural_sector_ratios.items()}

    return {
        "Total Energy": adjusted_total_energy,
        "Urban Usage": urban_usage,
        "Rural Usage": rural_usage,
        "Urban Distribution": urban_distribution,
        "Rural Distribution": rural_distribution,
    }

####################################
# create_full_feature_vector
####################################
def create_full_feature_vector(basic_features, temp, rainfall, humidity, population):
    full_vector = [0]*30
    full_vector[0] = basic_features[0]  # Date integer
    full_vector[1] = basic_features[1]  # Festival flag
    full_vector[2] = 1.0  # Placeholder
    full_vector[3] = 1.0  # Placeholder
    full_vector[4] = temp
    full_vector[5] = rainfall
    full_vector[6] = humidity
    full_vector[7] = population
    for i in range(8, 30):
        full_vector[i] = 1.0  # Padding
    return full_vector

####################################
# convert_to_builtin_type
####################################
def convert_to_builtin_type(obj):
    """Recursively convert np.float32 or dicts/lists of them into Python built-ins."""
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_builtin_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin_type(v) for v in obj]
    else:
        return obj