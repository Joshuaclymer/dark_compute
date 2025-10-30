import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Construction time in months (groundbreaking to operation at capacity)
construction_months = [
    12,
    42,     # Infineon Dresden: May 2023 → Fall 2026 (PLANNED)
    26,     # SK Hynix Yongin: Mar 2025 → May 2027 (PLANNED)
    36,     # TSMC Fab 18: Jan 2018 → all phases by 2021 (ACTUAL)
    42,     # TSMC Arizona: Mid 2021 → Early 2025 (ACTUAL)
    36,     # GlobalFoundries Fab 8: Jul 2009 → 2012 (ACTUAL)
    30,     # SK Hynix M16: Dec 2018 → mid-2021 production ramp (ACTUAL)
    24,     # Micron Fab 10: Apr 2018 → 2020 volume production (ACTUAL)
    26      # Samsung Pyeongtaek Line 1: May 2015 → Jul 2017 (ACTUAL)
]

# Multiply by 1.5x
construction_months_adjusted = [m * 1.5 for m in construction_months]

# Convert months to years
construction_years_adjusted = [m / 12 for m in construction_months_adjusted]

# Sort the data for CDF
sorted_data = np.sort(construction_years_adjusted)

# Calculate empirical CDF (probability values)
n = len(sorted_data)
cdf = np.arange(1, n + 1) / n

# Add point at origin for the empirical CDF
sorted_data_with_origin = np.concatenate([[0], sorted_data])
cdf_with_origin = np.concatenate([[0], cdf])

# Fit gamma distribution to the data
shape, loc, scale = stats.gamma.fit(construction_years_adjusted, floc=0)

# Generate points for the fitted gamma CDF
x_range = np.linspace(0, max(construction_years_adjusted) * 1.1, 1000)
fitted_cdf = stats.gamma.cdf(x_range, shape, loc=loc, scale=scale)

# Create the plot
plt.figure(figsize=(10, 6))
plt.step(sorted_data_with_origin, cdf_with_origin, where='post', linewidth=2,
         color='blue', label='Empirical CDF (1.5x adjustment)')
plt.scatter(sorted_data_with_origin, cdf_with_origin, color='blue', s=50, zorder=5)

# Plot fitted gamma distribution
plt.plot(x_range, fitted_cdf, 'r-', linewidth=2, alpha=0.7,
         label=f'Fitted Gamma (shape={shape:.2f}, scale={scale:.2f})')

# Add labels and title
plt.xlabel('Construction Time (Years)', fontsize=12)
plt.ylabel('Cumulative Probability', fontsize=12)
plt.title('Empirical CDF of Semiconductor Fab Construction Time (1.5x Adjustment)',
          fontsize=14, fontweight='bold')

# Set y-axis limits
plt.ylim(0, 1)
plt.xlim(left=0)

# Add grid for better readability
plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Add legend
plt.legend(loc='lower right', fontsize=10)

# Add horizontal lines at key probabilities
for prob in [0.25, 0.5, 0.75]:
    plt.axhline(y=prob, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

plt.tight_layout()
plt.savefig('construction_time_cdf.png', dpi=300, bbox_inches='tight')

print(f"Plot saved as 'construction_time_cdf.png'")
print(f"\nAdjusted Construction Time Statistics (1.5x multiplier):")
print(f"Number of data points: {len(construction_years_adjusted)}")
print(f"Mean: {np.mean(construction_years_adjusted):.2f} years")
print(f"Median: {np.median(construction_years_adjusted):.2f} years")
print(f"Min: {min(construction_years_adjusted):.2f} years")
print(f"Max: {max(construction_years_adjusted):.2f} years")
print(f"25th percentile: {np.percentile(construction_years_adjusted, 25):.2f} years")
print(f"75th percentile: {np.percentile(construction_years_adjusted, 75):.2f} years")
print(f"\nGamma Distribution Parameters:")
print(f"Shape (k): {shape:.4f}")
print(f"Scale (θ): {scale:.4f}")
print(f"Location: {loc:.4f}")
print(f"Fitted mean: {shape * scale:.2f} years")
print(f"Fitted std: {np.sqrt(shape) * scale:.2f} years")
