import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# Data from compute_vs_node_multiple_lines.py (excluding 6nm, 12nm, 130nm, and 180nm)
process_node = [3, 4, 5, 7, 10, 16, 22, 28, 40, 65, 90]

transistor_density = [215.6, 145.86, 137.6, 90.64, 60.3, 28.88, 16.50, 14.44, 7.22, 3.61, 1.80]

year_process_node_first_reached_high_volume_manufacturing = [
    2023,  # 3nm (TSMC N3E)
    2022,  # 4nm (TSMC N4)
    2020,  # 5nm (TSMC N5)
    2018,  # 7nm (TSMC N7)
    2017,  # 10nm (TSMC CLN10FF)
    2015,  # 16nm (TSMC 16nm FinFET Plus)
    2012,  # 22nm (Intel 22nm tri-gate)
    2011,  # 28nm (TSMC 28nm HKMG)
    2009,  # 40nm
    2006,  # 65nm
    2004,  # 90nm
]

# Create the plot
plt.figure(figsize=(10, 6))

# Fit a line to log(transistor_density) vs year
# log(density) = a * year + b, which means density = exp(b) * exp(a * year)
years_array = np.array(year_process_node_first_reached_high_volume_manufacturing)
density_array = np.array(transistor_density)
log_density = np.log(density_array)

# Linear fit in log space
coefficients = np.polyfit(years_array, log_density, 1)
a, b = coefficients  # a is slope, b is intercept

# Calculate R-squared
log_density_fit = a * years_array + b
residuals = log_density - log_density_fit
ss_res = np.sum(residuals**2)
ss_tot = np.sum((log_density - np.mean(log_density))**2)
r_squared = 1 - (ss_res / ss_tot)

# Generate fitted line
years_fit = np.linspace(min(years_array), max(years_array), 100)
density_fit = np.exp(a * years_fit + b)

# Plot transistor density over time
plt.scatter(year_process_node_first_reached_high_volume_manufacturing, transistor_density,
            alpha=0.6, s=150, color='blue', zorder=3)
plt.plot(year_process_node_first_reached_high_volume_manufacturing, transistor_density,
         "-", alpha=0.7, color='blue', linewidth=2, zorder=2)

# Calculate rate of change per year
rate_per_year = np.exp(a)

# Plot fitted line
plt.plot(years_fit, density_fit, '--', color='red', linewidth=2, zorder=2,
         label=f'Exponential fit: {rate_per_year:.2f}× per year (R² = {r_squared:.4f})')

# Add labels for each process node
for year, density, node in zip(year_process_node_first_reached_high_volume_manufacturing,
                                transistor_density, process_node):
    plt.annotate(f'{node}nm',
                xy=(year, density),
                xytext=(0, 10),  # Offset text slightly above the point
                textcoords='offset points',
                fontsize=10,
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='blue', alpha=0.7))

# Labels and title
plt.xlabel('Year of High Volume Manufacturing', fontsize=16)
plt.ylabel('Transistor Density (MTr/mm²)', fontsize=16)
plt.title('Transistor Density Over Time by Process Node', fontsize=18, pad=20)

# Add legend
plt.legend(loc='upper left', fontsize=12)

# Set log scale for y-axis
plt.yscale('log')

# Add grid for better readability
plt.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)

# Format tick labels
plt.tick_params(axis='both', labelsize=14)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('transistor_density_over_time.png', dpi=300, bbox_inches='tight')

print("Plot saved as 'transistor_density_over_time.png'")
print(f"Transistor density range: {min(transistor_density):.2f} - {max(transistor_density):.2f} MTr/mm²")
print(f"Year range: {min(year_process_node_first_reached_high_volume_manufacturing)} - {max(year_process_node_first_reached_high_volume_manufacturing)}")
