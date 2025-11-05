import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Read the CSV file
df = pd.read_csv('epoch_data.csv')

# Extract relevant columns
# Column 24: Number of transistors in million
# Column 25: Die Size in mm^2
# Column 3: Release year

# Filter for rows that have both transistor count and die size data
df_filtered = df[['Name of the hardware', 'Release year', 'Number of transistors in million', 'Die Size in mm^2']].copy()
df_filtered.columns = ['name', 'year', 'transistors_million', 'die_size']

# Remove rows with missing data
df_filtered = df_filtered.dropna()

# Calculate transistor density (millions of transistors per mm^2)
df_filtered['density'] = df_filtered['transistors_million'] / df_filtered['die_size']

# Calculate log of density for regression
df_filtered['log_density'] = np.log10(df_filtered['density'])

# Sort by year
df_filtered = df_filtered.sort_values('year')

print(f"Total data points: {len(df_filtered)}")
print(f"Year range: {df_filtered['year'].min()} to {df_filtered['year'].max()}")

# Perform linear regression on log-transformed data
slope, intercept, r_value, p_value, std_err = stats.linregress(df_filtered['year'], df_filtered['log_density'])

# Create the plot
plt.figure(figsize=(7, 5))

# Define color for points and regression line
point_color = '#1f77b4'  # Blue

# Plot the actual data points
plt.scatter(df_filtered['year'], df_filtered['density'], alpha=0.6, s=50, color=point_color, label='Actual Data')

# Add labels only for A100 and H100
for idx, row in df_filtered.iterrows():
    if 'A100' in row['name'] or 'H100' in row['name']:
        plt.annotate(row['name'],
                     (row['year'], row['density']),
                     fontsize=8,
                     alpha=0.8,
                     xytext=(5, 5),
                     textcoords='offset points')

# Plot the regression line (convert back from log scale)
years_range = np.linspace(df_filtered['year'].min(), df_filtered['year'].max(), 100)
log_regression_line = slope * years_range + intercept
regression_line = 10 ** log_regression_line
plt.plot(years_range, regression_line, color=point_color, linestyle='--', linewidth=2, label=f'Regression Line (R²={r_value**2:.3f})')

# Set y-axis to log scale
plt.yscale('log')

# Add labels
plt.xlabel('Year', fontsize=12)
plt.ylabel('Transistor Density (Million Transistors / mm²)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, which='both')

# Add regression equation to the plot
# Calculate annual rate of increase: 10^slope gives the multiplicative factor per year
annual_rate = 10 ** slope
equation_text = f'log₁₀(density) = {slope:.4f}x + {intercept:.4f}'
plt.text(0.05, 0.95, equation_text + f'\nAnnual increase: {annual_rate:.3f}x',
         transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('transistor_density_over_time.png', dpi=300, bbox_inches='tight')

print(f"\nRegression Statistics (on log-transformed data):")
print(f"Slope: {slope:.6f} (log₁₀(density) per year)")
print(f"Intercept: {intercept:.6f}")
print(f"R-squared: {r_value**2:.6f}")
print(f"P-value: {p_value:.6e}")
print(f"Standard Error: {std_err:.6f}")
print(f"\nAnnual rate of increase: {annual_rate:.3f}x")
print(f"This means density increases by 10x every {1/slope:.2f} years")
