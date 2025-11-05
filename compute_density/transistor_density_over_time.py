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

# Create the plot with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# --- First subplot: Transistor density vs Year (original) ---
years_array = np.array(year_process_node_first_reached_high_volume_manufacturing)
density_array = np.array(transistor_density)
log_density = np.log(density_array)

# Linear fit in log space (year vs density)
coefficients_year = np.polyfit(years_array, log_density, 1)
a_year, b_year = coefficients_year

# Calculate R-squared for year fit
log_density_fit_year = a_year * years_array + b_year
residuals_year = log_density - log_density_fit_year
ss_res_year = np.sum(residuals_year**2)
ss_tot_year = np.sum((log_density - np.mean(log_density))**2)
r_squared_year = 1 - (ss_res_year / ss_tot_year)

# Generate fitted line for year
years_fit = np.linspace(min(years_array), max(years_array), 100)
density_fit_year = np.exp(a_year * years_fit + b_year)

# Plot transistor density over time
ax1.scatter(year_process_node_first_reached_high_volume_manufacturing, transistor_density,
            alpha=0.6, s=150, color='blue', zorder=3)
ax1.plot(year_process_node_first_reached_high_volume_manufacturing, transistor_density,
         "-", alpha=0.7, color='blue', linewidth=2, zorder=2)

# Calculate rate of change per year
rate_per_year = np.exp(a_year)

# Plot fitted line
ax1.plot(years_fit, density_fit_year, '--', color='red', linewidth=2, zorder=2,
         label=f'Exponential fit: {rate_per_year:.2f}× per year (R² = {r_squared_year:.4f})')

# Add labels for each process node
for year, density, node in zip(year_process_node_first_reached_high_volume_manufacturing,
                                transistor_density, process_node):
    ax1.annotate(f'{node}nm',
                xy=(year, density),
                xytext=(0, 10),
                textcoords='offset points',
                fontsize=10,
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='blue', alpha=0.7))

ax1.set_xlabel('Year of High Volume Manufacturing', fontsize=16)
ax1.set_ylabel('Transistor Density (MTr/mm²)', fontsize=16)
ax1.set_title('Transistor Density Over Time by Process Node', fontsize=18, pad=20)
ax1.legend(loc='upper left', fontsize=12)
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
ax1.tick_params(axis='both', labelsize=14)

# --- Second subplot: Transistor density vs Process Node (NEW) ---
process_node_array = np.array(process_node)
log_process_node = np.log(process_node_array)

# Linear fit in log-log space (process node vs density)
# log(density) = a * log(node) + b, which means density = exp(b) * node^a
coefficients_node = np.polyfit(log_process_node, log_density, 1)
a_node, b_node = coefficients_node

# Calculate R-squared for process node fit
log_density_fit_node = a_node * log_process_node + b_node
residuals_node = log_density - log_density_fit_node
ss_res_node = np.sum(residuals_node**2)
ss_tot_node = np.sum((log_density - np.mean(log_density))**2)
r_squared_node = 1 - (ss_res_node / ss_tot_node)

# Generate fitted line for process node
process_node_fit = np.linspace(min(process_node_array), max(process_node_array), 100)
density_fit_node = np.exp(b_node) * process_node_fit**a_node

# Plot transistor density vs process node
ax2.scatter(process_node, transistor_density,
            alpha=0.6, s=150, color='green', zorder=3)
ax2.plot(process_node, transistor_density,
         "-", alpha=0.7, color='green', linewidth=2, zorder=2)

# Plot fitted line
ax2.plot(process_node_fit, density_fit_node, '--', color='red', linewidth=2, zorder=2,
         label=f'Power law fit: density = {np.exp(b_node):.2f} × node^{a_node:.2f} (R² = {r_squared_node:.4f})')

# Add labels for each process node with year
for node, density, year in zip(process_node, transistor_density,
                                year_process_node_first_reached_high_volume_manufacturing):
    ax2.annotate(f'{node}nm\n({year})',
                xy=(node, density),
                xytext=(0, 10),
                textcoords='offset points',
                fontsize=9,
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='green', alpha=0.7))

ax2.set_xlabel('Process Node (nm)', fontsize=16)
ax2.set_ylabel('Transistor Density (MTr/mm²)', fontsize=16)
ax2.set_title('Transistor Density vs Process Node', fontsize=18, pad=20)

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
