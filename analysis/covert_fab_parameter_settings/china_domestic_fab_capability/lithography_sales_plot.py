import matplotlib.pyplot as plt
import numpy as np

# DUV Immersion Systems Data
duv_start_year = 2006
duv_years = [2006, 2007, 2008, 2009, 2010, 2011]
duv_sales = [23, 35, 56, 35, 90, 101]
duv_years_relative = [year - duv_start_year for year in duv_years]

# EUV Systems Data
euv_start_year = 2019
euv_years = [2019, 2020, 2021, 2022, 2023, 2024]
euv_sales = [26, 31, 42, 40, 53, 44]
euv_years_relative = [year - euv_start_year for year in euv_years]

# Estimated PRC Rampup Data
prc_years_relative = [0, 5]
prc_sales = [20, 100]

# Calculate linear fit coefficients for PRC Rampup
# y = mx + b
slope = (prc_sales[1] - prc_sales[0]) / (prc_years_relative[1] - prc_years_relative[0])
intercept = prc_sales[0] - slope * prc_years_relative[0]

# Create the plot
plt.figure(figsize=(8, 5))

# Plot all datasets
plt.plot(duv_years_relative, duv_sales, marker='o', linewidth=2, markersize=8,
         label=f'DUV Immersion Systems (starting {duv_start_year})', color='#2E86AB')
plt.plot(euv_years_relative, euv_sales, marker='s', linewidth=2, markersize=8,
         label=f'EUV Systems (starting {euv_start_year})', color='#A23B72')
plt.plot(prc_years_relative, prc_sales, linestyle='--', linewidth=2,
         label=f'Estimated PRC Rampup (y = {slope:.1f}x + {intercept:.1f})', color='#F77F00')

# Customize the plot
plt.xlabel('Years After First High Volume Production', fontsize=12, fontweight='bold')
plt.ylabel('Lithography Scanners Sold by ASML', fontsize=12, fontweight='bold')
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3, linestyle='--')

# Set integer ticks on y-axis
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Add some spacing
plt.tight_layout()

# Save the plot
plt.savefig('lithography_sales_plot.png', dpi=300, bbox_inches='tight')
