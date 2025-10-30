import matplotlib.pyplot as plt
import numpy as np


capacity_wspm = [
    55000,      # VSMC Singapore (full capacity by 2029)
    31500,      # Infineon Dresden (average of 30,000-33,000)
    200000,     # SK Hynix Yongin 1st fab only (per 2021 reports)
    83000,      # TSMC Fab 18 (all 3 phases combined)
    20000,      # TSMC Arizona Fab 21
    33000,      # GlobalFoundries Fab 8 (400,000/year)
    62500,      # Intel Fab 42 (midpoint of 25,000-100,000)
    100000,     # SK Hynix M16
    140000,     # Micron Fab 10 (expansion capacity)
    150000,     # Samsung Pyeongtaek Line 1 only (not full 450k campus)
    5000
]

# Construction time in months (groundbreaking to operation at capacity)
construction_months = [
    60,     # VSMC Singapore: Dec 2024 → full capacity 2029 (PLANNED)
    42,     # Infineon Dresden: May 2023 → Fall 2026 (PLANNED)
    26,     # SK Hynix Yongin: Mar 2025 → May 2027 (PLANNED)
    36,     # TSMC Fab 18: Jan 2018 → all phases by 2021 (ACTUAL)
    42,     # TSMC Arizona: Mid 2021 → Early 2025 (ACTUAL)
    36,     # GlobalFoundries Fab 8: Jul 2009 → 2012 (ACTUAL)
    108,    # Intel Fab 42: 2011 → 2020 operation (ACTUAL, includes mothball period)
    30,     # SK Hynix M16: Dec 2018 → mid-2021 production ramp (ACTUAL)
    24,     # Micron Fab 10: Apr 2018 → 2020 volume production (ACTUAL)
    26,      # Samsung Pyeongtaek Line 1: May 2015 → Jul 2017 (ACTUAL)
    12,
]

# Convert months to years
construction_years = [m / 12 for m in construction_months]

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(capacity_wspm, construction_years, alpha=0.6, s=100, color='green')

# Add labels and title
plt.xlabel('Fab Production Capacity (Wafers per Month)', fontsize=12)
plt.ylabel('Construction Time (Years)', fontsize=12)
plt.title('Semiconductor Fab: Production Capacity vs Construction Time', fontsize=14, fontweight='bold')

# Set log scale for x-axis
plt.xscale('log')

# Set specific tick marks for better readability
# X-axis (capacity)
x_ticks = [5000, 10000, 20000, 50000, 100000, 200000, 500000]
plt.xticks(x_ticks, [f'{int(x/1000)}K' for x in x_ticks])

plt.ylim(bottom=0)
# Add grid for better readability
plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Add a trend line (in semi-log space)
log_capacity = np.log10(capacity_wspm)
z = np.polyfit(log_capacity, construction_years, 1)
p = np.poly1d(z)
x_range = np.logspace(np.log10(min(capacity_wspm)), np.log10(max(capacity_wspm)), 100)

plt.legend(loc='best', fontsize=10)
plt.tight_layout()
plt.savefig('capacity_vs_construction_time.png', dpi=300, bbox_inches='tight')

print(f"Plot saved as 'capacity_vs_construction_time.png'")
print(f"\nData points: {len(capacity_wspm)}")
print(f"Construction time range: {min(construction_years):.2f} - {max(construction_years):.2f} years")
print(f"Capacity range: {min(capacity_wspm):,} - {max(capacity_wspm):,} wafers/month")