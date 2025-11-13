# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
df = pd.read_csv('epoch_data.csv')

# Extract die size and TDP columns
# Remove rows with missing values for either column
data = df[['Die Size in mm^2', 'TDP in W']].dropna()

# Create the plot
plt.figure(figsize=(12, 8))
plt.scatter(data['Die Size in mm^2'], data['TDP in W'], alpha=0.6, s=100, edgecolors='black', linewidth=0.5)

# Add labels and title
plt.xlabel('Die Size (mm�)', fontsize=12)
plt.ylabel('Energy Consumption (TDP in W)', fontsize=12)
plt.title('Energy Consumption vs Die Size for GPUs/Accelerators', fontsize=14, fontweight='bold')

# Add grid for better readability
plt.grid(True, alpha=0.3, linestyle='--')

# Add a trend line
z = np.polyfit(data['Die Size in mm^2'], data['TDP in W'], 1)
p = np.poly1d(z)
x_trend = np.linspace(data['Die Size in mm^2'].min(), data['Die Size in mm^2'].max(), 100)
plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Linear fit: y={z[0]:.2f}x+{z[1]:.2f}')

plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig('energy_vs_die_size.png', dpi=300, bbox_inches='tight')
print(f"Plot saved as 'energy_vs_die_size.png'")
print(f"\nData summary:")
print(f"Number of chips with both die size and TDP data: {len(data)}")
print(f"Die size range: {data['Die Size in mm^2'].min():.1f} - {data['Die Size in mm^2'].max():.1f} mm�")
print(f"TDP range: {data['TDP in W'].min():.1f} - {data['TDP in W'].max():.1f} W")
print(f"Correlation coefficient: {data['Die Size in mm^2'].corr(data['TDP in W']):.3f}")

# Don't show plot interactively
# plt.show()
