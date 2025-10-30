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
dies_per_wafer = 28

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

fab_capacity_wspm = 1000  # wafers per month

# Calculate H100 equivalents per month
h100_equiv_per_month = [equiv * fab_capacity_wspm for equiv in h100_equiv_per_wafer]
h100_equiv_per_year = [equiv * 12 for equiv in h100_equiv_per_month]

# Create second scatter plot
plt.figure(figsize=(12, 7))

# Add scatter point for H100
plt.scatter(process_node[h100_index], h100_equiv_per_wafer[h100_index],
            alpha=0.6, s=100, color='green', zorder=3)

# Label key GPU points
plt.annotate('H100', (process_node[h100_index], h100_equiv_per_wafer[h100_index]),
            textcoords="offset points", xytext=(5, -20), ha='left', fontsize=14, alpha=0.8)


# Add labels
plt.xlabel('TSMC Process Node (nm)', fontsize=16)
plt.ylabel('H100 Equivalents per wafer (2022 architectures)', fontsize=16)

# Set log-log scale
plt.xscale('log')
plt.yscale('log')
plt.xlim(left=2, right=200)
plt.gca().invert_xaxis()

# Add line connecting all data points
plt.plot(process_node, h100_equiv_per_wafer, "-", alpha=0.7, color='green', linewidth=2, zorder=2)

# Add grid for better readability
plt.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)

# Set specific x-axis ticks for better readability
x_ticks2 = [3, 4, 5, 6, 7, 10, 12, 16, 22, 28, 40, 65, 90, 130, 180]
plt.xticks(x_ticks2, [str(x) for x in x_ticks2], fontsize=14, rotation=45)

# Add secondary x-axis for years
ax_month = plt.gca()
ax2_month = ax_month.twiny()
ax2_month.set_xscale('log')
ax2_month.set_xlim(ax_month.get_xlim())
ax2_month.set_xticks(x_ticks2)
ax2_month.set_xticklabels([str(year) for year in year_process_node_first_reached_high_volume_manufacturing],
                          fontsize=14, rotation=45)
ax2_month.set_xlabel('Year of High Volume Manufacturing', fontsize=16)

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

ax_month.yaxis.set_major_formatter(FuncFormatter(format_func2))
ax_month.tick_params(axis='y', labelsize=14)


# Add secondary y-axis for transistor density
ax_density = ax_month.twinx()
ax_density.scatter(process_node, transistor_density, alpha=0.0)  # Invisible scatter to set scale
ax_density.set_yscale('log')
ax_density.set_ylabel('Transistor Density (MTr/mm²)', fontsize=16, color='green')
ax_density.tick_params(axis='y', labelcolor='green', labelsize=14)
ax_density.set_ylim(min(transistor_density) * 0.8, max(transistor_density) * 1.2)

plt.tight_layout()
plt.savefig('h100_equiv_per_year_vs_process_node.png', dpi=300, bbox_inches='tight')

print(f"Plot saved as 'h100_equiv_per_year_vs_process_node.png'")
print(f"H100 equivalents per wafer range: {min(h100_equiv_per_wafer):.2f} - {max(h100_equiv_per_wafer):.2f}")