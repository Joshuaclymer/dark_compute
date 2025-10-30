import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# Node data for plotting

h100_node = 4
TPP_per_transistor_density = 644

process_node = [3, 4, 5, 6, 7, 10, 12, 16, 22, 28, 40, 65, 90, 130, 180]

transistor_density = [215.6, 145.86, 137.6, 106.96, 90.64, 60.3, 33.8, 28.88, 16.50, 14.44, 7.22, 3.61, 1.80, 0.90, 0.45]

year_process_node_first_reached_high_volume_manufacturing = [
    2023,  # 3nm (TSMC N3E)
    2022,  # 4nm (TSMC N4)
    2020,  # 5nm (TSMC N5)
    2020,  # 6nm (TSMC N6)
    2018,  # 7nm (TSMC N7)
    2017,  # 10nm (TSMC CLN10FF)
    2018,  # 12nm (TSMC 12nm FFN)
    2015,  # 16nm (TSMC 16nm FinFET Plus)
    2012,  # 22nm (Intel 22nm tri-gate)
    2011,  # 28nm (TSMC 28nm HKMG)
    2009,  # 40nm
    2006,  # 65nm
    2004,  # 90nm
    2001,  # 130nm
    1999   # 180nm
]

# Calculate H100 equivalents per wafer
# 12 inch wafer = 304.8mm diameter
# wafer_diameter = 304.8  # mm
# wafer_area = np.pi * (wafer_diameter / 2) ** 2  # mm²
# yield_rate = 0.80

# H100 specs
# h100_die_size = 814  # mm²
h100_tpp = 63328

# Assume all chips use H100 die size for comparison purposes
# die_size = h100_die_size

# Calculate TPP for each node based on transistor density
# TPP = transistor_density * TPP_per_transistor_density
tpp = [td * TPP_per_transistor_density for td in transistor_density]

# Calculate dies per wafer (accounting for yield)
dies_per_wafer = 50

# Calculate H100 equivalents per wafer for each node
# (dies per wafer) * (TPP per die) / (H100 TPP)
h100_equiv_per_wafer = [(dies_per_wafer * tpp_val) / h100_tpp for tpp_val in tpp]

# Label key GPU point indices
h100_index = process_node.index(4)
a100_index = process_node.index(7)
v100_index = process_node.index(12)

# ============================================================================
# H100 equivalents per month from a 10K WSPM fab
# ============================================================================

fab_capacity_wspm = 10000  # wafers per month

# Calculate H100 equivalents per month
h100_equiv_per_month = [equiv * fab_capacity_wspm for equiv in h100_equiv_per_wafer]
h100_equiv_per_year = [equiv * 12 for equiv in h100_equiv_per_month]

# Create second scatter plot
plt.figure(figsize=(12, 7))
plt.scatter(year_process_node_first_reached_high_volume_manufacturing, h100_equiv_per_year, alpha=0.6, s=100, color='green', zorder=3)

# Label each point with process node number
for i, (year, equiv, node) in enumerate(zip(year_process_node_first_reached_high_volume_manufacturing, h100_equiv_per_year, process_node)):
    plt.annotate(f'{node}nm', (year, equiv),
                textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, alpha=0.7)


# Add labels
plt.xlabel('Year of High Volume Manufacturing', fontsize=16)
plt.ylabel('H100 Equivalents per year\n(estimated assuming a 10K WPM fab)', fontsize=16)

# Set y-axis to log scale
plt.yscale('log')

# Add line connecting all data points
plt.plot(year_process_node_first_reached_high_volume_manufacturing, h100_equiv_per_year, "-", alpha=0.7, color='green', linewidth=2, zorder=2)

# Add grid for better readability
plt.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)

# Format y-axis with regular numbers
def format_func2(value, tick_number):
    if value >= 1000000:
        return f'{int(value/1000000)}M'
    elif value >= 1000:
        return f'{int(value/1000)}K'
    elif value >= 1:
        return f'{int(value)}'
    else:
        return f'{value:.2f}'

ax_month = plt.gca()
ax_month.yaxis.set_major_formatter(FuncFormatter(format_func2))
ax_month.tick_params(axis='y', labelsize=14)
ax_month.tick_params(axis='x', labelsize=14)

# Add horizontal line at 10K H100e per month
ax_month.axhline(y=87000, color='black', linestyle='--', linewidth=2, alpha=0.7, zorder=1)
ax_month.text(min(year_process_node_first_reached_high_volume_manufacturing), 87000 * 1.2, 'Divert 3% of new phones consumed by China',
             fontsize=14, color='black', ha='left', va='bottom')

# Add secondary y-axis for transistor density
ax_density = ax_month.twinx()
ax_density.scatter(year_process_node_first_reached_high_volume_manufacturing, transistor_density, alpha=0.0)  # Invisible scatter to set scale
ax_density.set_yscale('log')
ax_density.set_ylabel('Transistor Density (MTr/mm²)', fontsize=16, color='green')
ax_density.tick_params(axis='y', labelcolor='green', labelsize=14)
ax_density.set_ylim(min(transistor_density) * 0.8, max(transistor_density) * 1.2)

plt.tight_layout()
plt.savefig('h100_equiv_per_year_vs_process_node.png', dpi=300, bbox_inches='tight')

print(f"Plot saved as 'h100_equiv_per_year_vs_process_node.png'")
print(f"H100 equivalents per year (10K WSPM fab) range: {min(h100_equiv_per_year):.2f} - {max(h100_equiv_per_year):.2f}")