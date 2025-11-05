import matplotlib.pyplot as plt
import numpy as np

# Capacity in WSPM (or equivalent for SiC/flexible electronics)
capacity_wspm_construction = [
    55000,      # VSMC Singapore (target by 2029)
    200000,     # SK Hynix Yongin 1st fab only
    83000,      # TSMC Fab 18 (all 3 phases combined)
    60000,      # GlobalFoundries Fab 8
    62500,      # Intel Fab 42 (midpoint of 25,000-100,000) 
    100000,     # SK Hynix M16
    140000,     # Micron Fab 10 (expansion)
    150000,     # Samsung Pyeongtaek Line 1 only
    5000,       # SMIC Shenzhen (initial)
    80000,      # GlobalFoundries Fab 1 Module 3 Dresden (full campus)
    4000,       # Pragmatic Semiconductor FlexLogic 001 Durham
    4000,       # TSMC Fab 15 Phase 1
    20000,      # TSMC Fab 21 Phase 1 Arizona
    20000,      # SMIC Fab 8 S2 Shanghai
    30000,      # Sanan IC Changsha (6-inch SiC wafers)
    30000       # Sanan IC Xiamen
]

# Construction time in months (groundbreaking to initial operation)
construction_months = [
    30,     # VSMC Singapore: Dec 2024 → 2027 initial production (PLANNED)
    26,     # SK Hynix Yongin: Mar 2025 → May 2027 (PLANNED)
    24,     # TSMC Fab 18: Jan 2018 → early 2020 first phase (ACTUAL)
    36,     # GlobalFoundries Fab 8: Jul 2009 → 2012 (ACTUAL)
    24,     # Intel Fab 42: 2011 → 2013 construction only (ACTUAL, excluding 7-yr idle)
    30,     # SK Hynix M16: Nov 2018 → H2 2021 production (ACTUAL)
    24,     # Micron Fab 10: Apr 2018 → 2020 volume production (ACTUAL)
    26,     # Samsung Pyeongtaek Line 1: May 2015 → Jul 2017 (ACTUAL)
    12,     # SMIC Shenzhen: End 2016 → End 2017 (PLANNED)
    27,     # GlobalFoundries Fab 1 Module 3 Dresden: H2 2010 → end 2012 (ACTUAL)
    12,     # Pragmatic Semiconductor FlexLogic 001: 2017 → early 2018 (ACTUAL)
    11,     # TSMC Fab 15 Phase 1: July 2010 → mid-2011 (ACTUAL)
    16,     # TSMC Fab 21 Arizona: Mar 2021 → Jul 2022 building completion (ACTUAL)
    19,     # SMIC Fab 8 S2 Shanghai: Oct 2005 → July 2007 (ACTUAL)
    11,     # Sanan IC Changsha: July 2020 → June 2021 (ACTUAL)
    15      # Sanan IC Xiamen: 2014 → Oct 2015 estimate 12-18 months (ACTUAL)
]

# Country location for each fab
country_construction = [
    'Singapore',    # VSMC Singapore
    'South Korea',  # SK Hynix Yongin
    'Taiwan',       # TSMC Fab 18
    'USA',          # GlobalFoundries Fab 8
    'USA',          # Intel Fab 42
    'South Korea',  # SK Hynix M16
    'Singapore',    # Micron Fab 10
    'South Korea',  # Samsung Pyeongtaek Line 1
    'China',        # SMIC Shenzhen
    'Europe',       # GlobalFoundries Fab 1 Module 3 Dresden
    'Europe',       # Pragmatic Semiconductor FlexLogic 001 Durham
    'Taiwan',       # TSMC Fab 15 Phase 1
    'USA',          # TSMC Fab 21 Phase 1 Arizona
    'China',        # SMIC Fab 8 S2 Shanghai
    'China',        # Sanan IC Changsha
    'China'         # Sanan IC Xiamen
]


# Convert months to years
construction_years = [m / 12 for m in construction_months]

# ===== DATA FROM capacity_vs_workers.py =====
capacity_wspm_workers = [83000, 20000, 62500, 22500, 45000, 70000, 60000]
construction_workers = [23500, 11000, 4000, 7000, 7000, 8000, 8500]

# Country location for each fab
country_workers = [
    'Taiwan',       # TSMC Fab 18
    'USA',          # TSMC Arizona Fab 21
    'USA',          # Intel Fab 42
    'USA',          # Intel Ohio
    'Europe',      # Intel Magdeburg
    'USA',          # Samsung Taylor
    'USA'           # GlobalFoundries Fab 8
]

# Define color mapping for countries
country_colors = {
    'USA': '#1f77b4',        # blue
    'South Korea': '#ff7f0e', # orange
    'Taiwan': '#2ca02c',      # green
    'Singapore': '#d62728',   # red
    'Europe': '#9467bd',     # purple
    'China': '#8c564b'        # brown
}

# Calculate regression line in semi-log space (original data)
log_capacity = np.log10(capacity_wspm_construction)
z = np.polyfit(log_capacity, construction_years, 1)
p = np.poly1d(z)
x_range_log = np.logspace(np.log10(min(capacity_wspm_construction)), np.log10(max(capacity_wspm_construction)), 100)
y_range_log = p(np.log10(x_range_log))

# Calculate regression line for transformed data (1.5x construction time) - "with concealment"
construction_years_transformed = [t * 1.5 for t in construction_years]
z_concealment = np.polyfit(log_capacity, construction_years_transformed, 1)
p_concealment = np.poly1d(z_concealment)
y_range_log_concealment = p_concealment(np.log10(x_range_log))

# ===== FIGURE 1: Original plot (without adjusted line) =====
fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))

# Plot each country with its own color
for country in set(country_construction):
    mask = [c == country for c in country_construction]
    capacities = [cap for cap, m in zip(capacity_wspm_construction, mask) if m]
    times = [time for time, m in zip(construction_years, mask) if m]
    ax1.scatter(capacities, times, alpha=0.6, s=50,
                color=country_colors[country], label=country)

ax1.set_xlabel('Fab Production Capacity (Wafers per Month)', fontsize=12)
ax1.set_ylabel('Construction Time (Years)', fontsize=12)

# Set log scale for x-axis
ax1.set_xscale('log')

# Set specific tick marks for better readability
x_ticks = [5000, 10000, 20000, 50000, 100000, 200000, 500000]
ax1.set_xticks(x_ticks)
ax1.set_xticklabels([f'{int(x/1000)}K' for x in x_ticks])

ax1.set_ylim(bottom=0)
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Add regression line with parameters
slope = z[0]
intercept = z[1]
ax1.plot(x_range_log, y_range_log, "--", alpha=0.5, color='gray', linewidth=2,
         label=f'Fit: y = {slope:.3f}*log10(x) + {intercept:.3f}')

ax1.legend(loc='best', fontsize=8)

plt.tight_layout()
plt.savefig('construction_time_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# ===== FIGURE 2: Plot with adjusted line =====
fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))

# Plot original points in grey
ax2.scatter(capacity_wspm_construction, construction_years, alpha=0.6, s=50,
            color='grey', label='Original data')

# Plot transformed points (1.5x construction time) in red
ax2.scatter(capacity_wspm_construction, construction_years_transformed, alpha=0.6, s=50,
            color='red', label='1.5x construction time')

ax2.set_xlabel('Fab Production Capacity (Wafers per Month)', fontsize=12)
ax2.set_ylabel('Construction Time (Years)', fontsize=12)

# Set log scale for x-axis
ax2.set_xscale('log')

# Set specific tick marks for better readability
ax2.set_xticks(x_ticks)
ax2.set_xticklabels([f'{int(x/1000)}K' for x in x_ticks])

# Calculate y-limit to show adjusted line up to x=100K
y_at_100k_concealment = p_concealment(np.log10(100000))
y_upper_limit = max(y_at_100k_concealment * 1.1, max(construction_years_transformed) * 1.1)  # 10% margin
ax2.set_ylim(bottom=0, top=y_upper_limit)
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Add original regression line with parameters
slope = z[0]
intercept = z[1]
ax2.plot(x_range_log, y_range_log, "--", alpha=0.5, color='gray', linewidth=2,
         label=f'Fit: y = {slope:.3f}*log10(x) + {intercept:.3f}')

# Add concealment regression line (fitted to 1.5x data)
slope_concealment = z_concealment[0]
intercept_concealment = z_concealment[1]
ax2.plot(x_range_log, y_range_log_concealment, "-", alpha=0.7, color='red', linewidth=2,
         label=f'With concealment: y = {slope_concealment:.3f}*log10(x) + {intercept_concealment:.3f}')

ax2.legend(loc='best', fontsize=8)

plt.tight_layout()
plt.savefig('construction_time_plot_adjusted.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Plot saved as 'construction_time_plot.png'")
print(f"Plot with adjusted line saved as 'construction_time_plot_adjusted.png'")
print(f"\nConstruction Time:")
print(f"  Data points: {len(capacity_wspm_construction)}")
print(f"  Construction time range: {min(construction_years):.2f} - {max(construction_years):.2f} years")
print(f"  Capacity range: {min(capacity_wspm_construction):,} - {max(capacity_wspm_construction):,} wafers/month")
