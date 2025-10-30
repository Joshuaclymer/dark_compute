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
dies_per_wafer = 28 # going off of NVIDIA numbers: https://www.trendforce.com/news/2024/03/18/news-tsmc-boosts-investment-in-advanced-packaging-with-ntd-500-billion-plan-to-build-six-plants-in-chiayi-science-park/

# Calculate H100 equivalents per wafer for each node
# (dies per wafer) * (TPP per die) / (H100 TPP)
h100_equiv_per_wafer = [(dies_per_wafer * tpp_val) / h100_tpp for tpp_val in tpp]

# Label key GPU point indices
h100_index = process_node.index(4)
a100_index = process_node.index(7)
v100_index = process_node.index(12)

# ============================================================================
# H100 equivalents per month from fabs with different capacities
# ============================================================================

fab_capacities_wspm = [5000, 50000, 500000]  # wafers per month
colors = ['green', 'blue', 'red']
labels = ['5K wafers per month, hundreds of workers (minifab)', '50K wafers per month, thousands of workers (typical fab)', '500K wafers per month, tens of thousands of workers (megafab)']

# Create second scatter plot
plt.figure(figsize=(12, 7))

# Plot each fab capacity
for fab_capacity, color, label in zip(fab_capacities_wspm, colors, labels):
    # Calculate H100 equivalents per month
    h100_equiv_per_month = [equiv * fab_capacity for equiv in h100_equiv_per_wafer]
    h100_equiv_per_year = [equiv * 12 for equiv in h100_equiv_per_month]

    plt.scatter(process_node, h100_equiv_per_year, alpha=0.6, s=100, color=color, zorder=3, label=label)
    plt.plot(process_node, h100_equiv_per_year, "-", alpha=0.7, color=color, linewidth=2, zorder=2)

# Add labels
plt.xlabel('TSMC Process Node (nm)', fontsize=16)
plt.ylabel('H100 Equivalents per year', fontsize=16)
plt.legend(loc='best', fontsize=12)

# Set log-log scale
plt.xscale('log')
plt.yscale('log')
plt.xlim(left=2, right=200)
plt.gca().invert_xaxis()

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

# Add horizontal line at 10K H100e per month
ax_month.axhline(y=87000, color='black', linestyle='--', linewidth=2, alpha=0.7, zorder=1)
ax_month.text(2.1, 87000 * 1.2, 'Divert 3% of new phones consumed by China',
             fontsize=14, color='black', ha='right', va='bottom')

# Add horizontal line for projected compute produced in 2027
ax_month.axhline(y=60000000, color='black', linestyle='--', linewidth=2, alpha=0.7, zorder=1)
ax_month.text(2.1, 60000000 * 1.2, 'Projected 2027 production',
             fontsize=14, color='black', ha='right', va='bottom')

# Add annotation at 28nm on the mini fab line
nm_28_index = process_node.index(28)
minifab_capacity = fab_capacities_wspm[0]  # 5000 wafers per month
nm_28_y_value = h100_equiv_per_wafer[nm_28_index] * minifab_capacity * 12
ax_month.annotate('Hypothetical covert PRC mini fab',
                 xy=(28, nm_28_y_value),
                 xytext=(28, nm_28_y_value * 3),
                 fontsize=12,
                 color='green',
                 ha='center',
                 va='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='green', alpha=0.8),
                 arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

# Add annotation at 90nm on the mini fab line
nm_90_index = process_node.index(90)
nm_90_y_value = h100_equiv_per_wafer[nm_90_index] * minifab_capacity * 12
ax_month.annotate("China's current domestic lithography",
                 xy=(90, nm_90_y_value),
                 xytext=(90, nm_90_y_value / 3),
                 fontsize=12,
                 color='green',
                 ha='center',
                 va='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='green', alpha=0.8),
                 arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

# Add secondary y-axis for transistor density
ax_density = ax_month.twinx()
ax_density.scatter(process_node, transistor_density, alpha=0.0)  # Invisible scatter to set scale
ax_density.set_yscale('log')
ax_density.set_ylabel('Transistor Density (MTr/mm²)', fontsize=16)
ax_density.tick_params(axis='y', labelsize=14)
ax_density.set_ylim(min(transistor_density) * 0.8, max(transistor_density) * 1.2)

plt.tight_layout()
plt.savefig('h100_equiv_per_year_vs_process_node_multiple.png', dpi=300, bbox_inches='tight')

print(f"Plot saved as 'h100_equiv_per_year_vs_process_node.png'")
print(f"H100 equivalents per year (5K WSPM fab) range: {min(h100_equiv_per_year):.2f} - {max(h100_equiv_per_year):.2f}")