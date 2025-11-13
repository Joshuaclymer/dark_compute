# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the main data
df = pd.read_csv('epoch_data.csv')

# Read additional older chips
try:
    additional = pd.read_csv('additional_older_chips.csv', sep='\t')

    # Extrapolate transistor counts based on year
    # First fit model on existing data
    year_transistor_data = df[['Release year', 'Number of transistors in million']].dropna()
    log_transistors = np.log10(year_transistor_data['Number of transistors in million'])
    years = year_transistor_data['Release year']
    z = np.polyfit(years, log_transistors, 1)

    # Clean up the additional data
    additional['Power (W)'] = additional['Power (W)'].str.replace('~', '').astype(float)
    additional['FP32 Performance (TFLOPS)'] = additional['FP32 Performance (TFLOPS)'].str.replace('~', '').str.replace('*', '').astype(float)

    # Predict transistor counts for older chips
    additional['Number of transistors in million'] = 10**(z[0] * additional['Year'] + z[1])

    # Extrapolate die sizes based on year
    die_size_data = df[['Release year', 'Die Size in mm^2']].dropna()
    log_die_size = np.log10(die_size_data['Die Size in mm^2'])
    years_die = die_size_data['Release year']
    z_die = np.polyfit(years_die, log_die_size, 1)
    additional['Die Size in mm^2'] = 10**(z_die[0] * additional['Year'] + z_die[1])

    # Convert to match main dataframe format
    additional['TDP in W'] = additional['Power (W)']
    additional['FP32 Performance (FLOP/s)'] = additional['FP32 Performance (TFLOPS)'] * 1e12
    additional['FP16 Performance (FLOP/s)'] = np.nan  # These old chips don't have FP16
    additional['Name of the hardware'] = additional['Chip Name']
    additional['Release year'] = additional['Year']

    # Combine with main data
    df = pd.concat([df, additional[['Number of transistors in million', 'Die Size in mm^2', 'TDP in W', 'FP32 Performance (FLOP/s)',
                                     'FP16 Performance (FLOP/s)', 'Name of the hardware', 'Release year']]], ignore_index=True)
    print(f"Added {len(additional)} older chips with extrapolated transistor counts")
except FileNotFoundError:
    print("No additional_older_chips.csv found, using only main dataset")

# Extract relevant columns including release year and die size
# Use FP16 Performance if available, otherwise use FP32 Performance
data = df[['Number of transistors in million', 'Die Size in mm^2', 'TDP in W', 'FP16 Performance (FLOP/s)', 'Name of the hardware', 'Release year']].copy()

# Fill missing FP16 with FP32 values where available
if 'FP32 Performance (FLOP/s)' in df.columns:
    data['FP16 Performance (FLOP/s)'].fillna(df['FP32 Performance (FLOP/s)'], inplace=True)

# Drop rows with any missing values
data_clean = data.dropna()

# Calculate transistor density (transistors per mm²)
data_clean['Transistor Density (M/mm^2)'] = data_clean['Number of transistors in million'] / data_clean['Die Size in mm^2']

# Remove T4 data point (suspected to be incorrect)
data_clean = data_clean[~data_clean['Name of the hardware'].str.contains('T4', na=False)]

# Calculate energy efficiency: Watts per TPP (lower is better)
# TPP (Tera-Parameter-Passes) = 2 × MacTOPS × bit_length
# For FP16: bit_length = 16, MacTOPS = FLOP/s (since 1 MAC = 2 FLOPs for FP16)
# For FP32: bit_length = 32, MacTOPS = FLOP/s / 2
# So: TPP = 2 × (FLOP/s / 2) × bit_length = FLOP/s × bit_length

# Determine bit length: use 16 for FP16, 32 for FP32
data_clean['bit_length'] = 16  # Default to FP16
# Mark chips that are using FP32 (those where original FP16 was NaN)
fp16_col = df['FP16 Performance (FLOP/s)']
data_clean['bit_length'] = data_clean.index.map(lambda idx: 16 if pd.notna(fp16_col.loc[idx]) else 32)

# Calculate TPP in operations per second, then convert to Tera
# TPP = FLOP/s × bit_length (in parameter passes per second)
data_clean['TPP (TPP/s)'] = data_clean['FP16 Performance (FLOP/s)'] * data_clean['bit_length']

# W/TPP = (TDP in W / TPP/s) * 1e12
data_clean['Energy per TPP (W/TPP)'] = (data_clean['TDP in W'] / data_clean['TPP (TPP/s)']) * 1e12

print(f"Dataset summary:")
print(f"Total chips with complete data: {len(data_clean)}")
print(f"\nEnergy Efficiency Statistics (W/TPP):")
print(f"  Min (most efficient): {data_clean['Energy per TPP (W/TPP)'].min():.6f} W/TPP")
print(f"  Max (least efficient): {data_clean['Energy per TPP (W/TPP)'].max():.6f} W/TPP")
print(f"  Mean: {data_clean['Energy per TPP (W/TPP)'].mean():.6f} W/TPP")
print(f"  Median: {data_clean['Energy per TPP (W/TPP)'].median():.6f} W/TPP")

# Find most and least efficient chips
most_efficient = data_clean.loc[data_clean['Energy per TPP (W/TPP)'].idxmin()]
least_efficient = data_clean.loc[data_clean['Energy per TPP (W/TPP)'].idxmax()]

print(f"\nMost efficient chip: {most_efficient['Name of the hardware']}")
print(f"  {most_efficient['Energy per TPP (W/TPP)']:.6f} W/TPP")
print(f"\nLeast efficient chip: {least_efficient['Name of the hardware']}")
print(f"  {least_efficient['Energy per TPP (W/TPP)']:.6f} W/TPP")

# Create the plot
plt.figure(figsize=(9, 5))

# Separate chips by FP precision only
fp16_mask = data_clean['bit_length'] == 16
fp32_mask = data_clean['bit_length'] == 32

# Plot FP16 chips in red
plt.scatter(data_clean[fp16_mask]['Transistor Density (M/mm^2)'],
           data_clean[fp16_mask]['Energy per TPP (W/TPP)'],
           alpha=0.6, s=100, edgecolors='black', linewidth=0.5, c='red',
           label='FP16', zorder=2)

# Plot FP32 chips in blue (includes both pre and post Dennard)
plt.scatter(data_clean[fp32_mask]['Transistor Density (M/mm^2)'],
           data_clean[fp32_mask]['Energy per TPP (W/TPP)'],
           alpha=0.6, s=100, edgecolors='black', linewidth=0.5, c='blue',
           label='FP32', zorder=1)

# Add labels and title
plt.xlabel('Transistor Density (M transistors/mm²)', fontsize=12)
plt.ylabel('Energy per TPP (W/TPP)', fontsize=12)
plt.title('Energy Efficiency vs Transistor Density\n(Lower is more efficient)', fontsize=14, fontweight='bold')

# Use log scale for both axes
plt.xscale('log')
plt.yscale('log')

# Add two separate regression lines (fit in log-log space)
# Exclude consumer GPU outliers: transistor count > 1000M and W/TPP > 40
outlier_mask = (data_clean['Number of transistors in million'] > 1000) & (data_clean['Energy per TPP (W/TPP)'] > 40)

# Power law for BEFORE end of Dennard scaling (≤2006)
pre_dennard_regression = data_clean[(data_clean['Release year'] <= 2006) & ~outlier_mask]
if len(pre_dennard_regression) > 1:
    log_density_pre = np.log10(pre_dennard_regression['Transistor Density (M/mm^2)'])
    log_efficiency_pre = np.log10(pre_dennard_regression['Energy per TPP (W/TPP)'])
    z_pre = np.polyfit(log_density_pre, log_efficiency_pre, 1)
    p_pre = np.poly1d(z_pre)

    x_trend_pre = np.logspace(np.log10(pre_dennard_regression['Transistor Density (M/mm^2)'].min()),
                               np.log10(pre_dennard_regression['Transistor Density (M/mm^2)'].max()), 100)
    y_trend_pre = 10**(p_pre(np.log10(x_trend_pre)))
    plt.plot(x_trend_pre, y_trend_pre, "--", alpha=0.8, linewidth=2, color='darkblue',
             label=f'Before Dennard end: y ∝ x^{z_pre[0]:.3f}')

# Power law for AFTER end of Dennard scaling (>2006)
post_dennard_regression = data_clean[(data_clean['Release year'] > 2006) & ~outlier_mask]
if len(post_dennard_regression) > 1:
    log_density_post = np.log10(post_dennard_regression['Transistor Density (M/mm^2)'])
    log_efficiency_post = np.log10(post_dennard_regression['Energy per TPP (W/TPP)'])
    z_post = np.polyfit(log_density_post, log_efficiency_post, 1)
    p_post = np.poly1d(z_post)

    x_trend_post = np.logspace(np.log10(post_dennard_regression['Transistor Density (M/mm^2)'].min()),
                                np.log10(post_dennard_regression['Transistor Density (M/mm^2)'].max()), 100)
    y_trend_post = 10**(p_post(np.log10(x_trend_post)))
    plt.plot(x_trend_post, y_trend_post, "--", alpha=0.8, linewidth=2, color='darkred',
             label=f'After Dennard end: y ∝ x^{z_post[0]:.3f}')

plt.legend()

# Add grid for better readability
plt.grid(True, alpha=0.3, linestyle='--', which='both')

# Don't annotate most efficient point
# (removed AMD Radeon label)

# Don't annotate the least efficient (GeForce3 Ti 500)
# plt.annotate(least_efficient['Name of the hardware'],
#             xy=(least_efficient['Number of transistors in million'], least_efficient['Energy per TPP (W/TPP)']),
#             xytext=(10, -15), textcoords='offset points',
#             bbox=dict(boxstyle='round,pad=0.5', fc='orange', alpha=0.7),
#             fontsize=8, ha='left')

# Annotate H100
h100_data = data_clean[data_clean['Name of the hardware'].str.contains('H100', na=False)]
if len(h100_data) > 0:
    h100 = h100_data.iloc[0]
    plt.annotate('H100',
                xy=(h100['Transistor Density (M/mm^2)'], h100['Energy per TPP (W/TPP)']),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                fontsize=7, ha='left')

# Annotate consumer GPUs with low FP16 (the red outliers at the top)
# These are GTX 1080 Ti and similar gaming cards with crippled FP16
low_fp16_outliers = data_clean[(data_clean['bit_length'] == 16) &
                                (data_clean['Energy per TPP (W/TPP)'] > 50) &
                                (data_clean['Release year'] > 2006)]
if len(low_fp16_outliers) > 0:
    # Pick one representative point for the label
    representative = low_fp16_outliers.iloc[0]
    plt.annotate('Consumer GPUs with\nlow FP16 performance',
                xy=(representative['Transistor Density (M/mm^2)'], representative['Energy per TPP (W/TPP)']),
                xytext=(10, -25), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightcoral', alpha=0.7),
                fontsize=7, ha='left')

# Add vertical line and label for end of Dennard Scaling (2006)
# Estimate transistor density at 2006 using our model
year_density_data = df[['Release year', 'Number of transistors in million', 'Die Size in mm^2']].dropna()
year_density_data['Transistor Density (M/mm^2)'] = year_density_data['Number of transistors in million'] / year_density_data['Die Size in mm^2']
year_density_clean = year_density_data[['Release year', 'Transistor Density (M/mm^2)']].dropna()
log_density = np.log10(year_density_clean['Transistor Density (M/mm^2)'])
years = year_density_clean['Release year']
z_year = np.polyfit(years, log_density, 1)
density_2006 = 10**(z_year[0] * 2006 + z_year[1])

# Draw vertical line
y_min, y_max = plt.ylim()
plt.axvline(x=density_2006, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, zorder=0)

# Add text label (positioned lower to avoid intersecting top)
plt.text(density_2006, y_max * 0.4, 'End of\nDennard Scaling\n(2006)',
         ha='center', va='bottom', fontsize=7,
         bbox=dict(boxstyle='round,pad=0.5', fc='lightgray', alpha=0.7))

plt.tight_layout()

# Save the plot
plt.savefig('energy_efficiency_vs_transistors.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved as 'energy_efficiency_vs_transistors.png'")

# Also print correlation
correlation = data_clean['Number of transistors in million'].corr(data_clean['Energy per TPP (W/TPP)'])
print(f"\nCorrelation between transistor count and energy per TPP: {correlation:.3f}")

# Don't show plot interactively
# plt.show()
