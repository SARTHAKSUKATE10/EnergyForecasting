import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# ------------------------------------------------------------------
# 1. Setup Logging
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------------------------------------------------------
# 2. Load Dataset
# ------------------------------------------------------------------
DATASET_PATH = r"C:\Users\jrsar\OneDrive\Desktop\FinalYearProject\backend\ML_Models\data\sectorwise_energy_updated.csv"


if not os.path.exists(DATASET_PATH):
    logging.error(f"Dataset not found: {DATASET_PATH}")
    exit(1)

df = pd.read_csv(DATASET_PATH)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month

# Select the energy usage column
energy_col = "Total Usage (kWh)"  

# Compute monthly average usage across all years
monthly_avg = df.groupby(["year", "month"])[energy_col].mean().reset_index()

# Pivot to get yearly trends for each month
pivot_table = monthly_avg.pivot(index="month", columns="year", values=energy_col)

# ------------------------------------------------------------------
# 3. Forecast Future Energy Usage (Oct 2024 - Dec 2025)
# ------------------------------------------------------------------
# Compute average monthly energy usage from previous years
monthly_trend = pivot_table.mean(axis=1)  # Average across all past years

# Generate future months
future_months = list(range(10, 13)) + list(range(1, 13))  # Oct-Dec 2024 + Jan-Dec 2025
future_years = [2024] * 3 + [2025] * 12

# Predict energy usage using historical trends
predicted_usage = [monthly_trend[m] for m in future_months]

# ------------------------------------------------------------------
# 4. Plot Historical vs. Predicted Data
# ------------------------------------------------------------------
plt.figure(figsize=(12, 6))

# Plot historical data
for yr in pivot_table.columns:
    plt.plot(pivot_table.index, pivot_table[yr], marker='o', linestyle='-', label=f"Historical {yr}")

# Plot predictions
plt.plot(future_months, predicted_usage, marker='o', linestyle='--', color='red', linewidth=2, label="Predicted 2024-2025")

plt.xlabel("Month")
plt.ylabel("Average Energy Usage (kWh)")
plt.title("Historical vs. Predicted Energy Usage (Oct 2024 - Dec 2025)")
plt.xticks(future_months, ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.grid(True)
plt.legend()
plt.show()

logging.info("✅ Prediction completed using historical trends.")
