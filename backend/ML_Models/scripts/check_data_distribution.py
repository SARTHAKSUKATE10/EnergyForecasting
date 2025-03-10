import numpy as np
import matplotlib.pyplot as plt

# Load the target (y) data for both sectors
y_rural = np.load("../data/y_Rural_Usage_(kWh).npy")
y_urban = np.load("../data/y_Urban_Usage_(kWh).npy")

# Plot histograms
plt.figure(figsize=(10, 5))
plt.hist(y_rural, bins=50, alpha=0.7, label="Rural Usage", color="red")
plt.hist(y_urban, bins=50, alpha=0.7, label="Urban Usage", color="blue")
plt.legend()
plt.xlabel("Energy Usage (kWh)")
plt.ylabel("Frequency")
plt.title("Distribution of Energy Usage")
plt.show()
