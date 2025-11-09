import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Read the CSV file
df = pd.read_csv('epoch_data.csv')

# Print column names to verify
print("Available columns:")
print(df.columns.tolist())

# Performance columns (in operations per second) - ordered from lowest to highest bit length
# TPP = 2 x MacTOPS x bit_length
# We'll use different performance metrics available in the data
perf_columns_ordered = [
    ('INT4 Tensor', 'INT4 Tensor Performance (OP/s)', 4),
    ('INT8 Tensor', 'INT8 Tensor Performance (OP/s)', 8),
    ('FP8 Tensor', 'FP8 Tensor Performance (FLOP/s)', 8),
    ('FP16', 'FP16 Performance (FLOP/s)', 16),
    ('FP16 Tensor', 'FP16 Tensor Performance (FLOP/s)', 16),
    ('INT16', 'INT16 Performance (OP/s)', 16),
    ('FP/TF32 Tensor', 'FP/TF32 Tensor Performance (FLOP/s)', 32),
    ('FP32', 'FP32 Performance (FLOP/s)', 32),
    ('FP64', 'FP64 Performance (FLOP/s)', 64),
]

# Die size column
die_size_col = 'Die Size in mm^2'

# Function to get the lowest bit length TPP for a chip
def get_lowest_bitlength_tpp(row):
    """Returns (TPP, bit_length, perf_type) for the lowest bit length available for this chip"""
    # First, get FP32 performance for reference
    fp32_perf = row['FP32 Performance (FLOP/s)']

    for perf_type, perf_col, bit_length in perf_columns_ordered:
        perf_value = row[perf_col]
        if pd.notna(perf_value) and perf_value > 0:
            # Filter out FP16 values that are suspiciously low (less than 10% of FP32)
            # This removes Pascal-era GPUs with crippled FP16
            if perf_type == 'FP16' and pd.notna(fp32_perf) and fp32_perf > 0:
                if perf_value < 0.1 * fp32_perf:
                    continue  # Skip this FP16 value, it's artificially low

            # Calculate TPP = 2 x MacTOPS x bit_length
            tpp = 2 * perf_value * bit_length / 1e12  # TPP in Tera-ops
            return tpp, bit_length, perf_type
    return None, None, None

# Create single figure
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

# Single color for all chips
chip_color = '#1f77b4'  # Blue

# Get all necessary columns
columns_needed = ['Name of the hardware', 'Release year', die_size_col] + [col for _, col, _ in perf_columns_ordered]
valid_data = df[columns_needed].dropna(subset=[die_size_col, 'Release year'])

# Store results for summary statistics
results = []

# Process each chip
for i, row in valid_data.iterrows():
    hardware_name = row['Name of the hardware']
    year = row['Release year']
    die_size = row[die_size_col]

    # Get lowest bit length TPP
    tpp, bit_length, perf_type = get_lowest_bitlength_tpp(row)

    if tpp is None:
        continue

    # Calculate TPP per die area
    tpp_per_area = tpp / die_size

    # Store for statistics
    results.append({
        'name': hardware_name,
        'year': year,
        'tpp_per_area': tpp_per_area,
        'bit_length': bit_length,
        'perf_type': perf_type
    })

# Convert results to DataFrame for analysis
results_df = pd.DataFrame(results)

# Find state-of-the-art chips (chips that achieve higher TPP/Die Area than ALL chips from previous years
# AND are the best in their own year)
sota_chips = []

for year in sorted(results_df['year'].unique()):
    # Get all chips from this year
    chips_this_year = results_df[results_df['year'] == year]

    # Get all chips from previous years
    chips_previous_years = results_df[results_df['year'] < year]

    # Find the max TPP/Die Area from previous years (0 if no previous years)
    if len(chips_previous_years) > 0:
        max_tpp_previous = chips_previous_years['tpp_per_area'].max()
    else:
        max_tpp_previous = 0

    # Find the max TPP/Die Area this year
    max_tpp_this_year = chips_this_year['tpp_per_area'].max()

    # Only include chips that:
    # 1. Beat all previous years
    # 2. Are the best (or tied for best) this year
    if max_tpp_this_year > max_tpp_previous:
        sota_this_year = chips_this_year[chips_this_year['tpp_per_area'] == max_tpp_this_year]
        sota_chips.extend(sota_this_year.index.tolist())

sota_df = results_df.loc[sota_chips]

# Plot all chips in blue
for idx, row in results_df.iterrows():
    ax.scatter(row['year'], row['tpp_per_area'],
              c=chip_color, marker='o',
              s=50, alpha=0.7, edgecolors='black', linewidth=0.5, zorder=2)

# Highlight state-of-the-art chips in green
sota_color = '#2ca02c'  # Green
for idx, row in sota_df.iterrows():
    ax.scatter(row['year'], row['tpp_per_area'],
              c=sota_color, marker='o',
              s=50, alpha=0.9, edgecolors='black', linewidth=0.8, zorder=3)

    # Add labels for state-of-the-art chips
    ax.annotate(row['name'], (row['year'], row['tpp_per_area']),
               fontsize=7, alpha=0.9,
               xytext=(5, 5), textcoords='offset points')

# Add regression line through state-of-the-art chips
if len(sota_df) > 1:
    # Perform linear regression in log space (since y-axis is log scale)
    log_tpp = np.log10(sota_df['tpp_per_area'])
    years = sota_df['year'].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(years, log_tpp)

    # Calculate annual growth multiplier (10^slope gives the multiplier per year)
    annual_multiplier = 10 ** slope

    # Generate line for plotting
    year_range = np.linspace(years.min(), years.max(), 100)
    log_tpp_fit = slope * year_range + intercept
    tpp_fit = 10 ** log_tpp_fit

    ax.plot(year_range, tpp_fit, color=sota_color, linewidth=2,
            linestyle='--', label=f'State-of-the-art trend ({annual_multiplier:.2f}× per year, R²={r_value**2:.3f})',
            zorder=4, alpha=0.8)

ax.set_xlabel('Release Year', fontsize=12)
ax.set_ylabel('TPP/Die Area (Tera-ops/mm²)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig('tpp_per_die_area.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'tpp_per_die_area.png'")

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS (Using Lowest Bit Length per Chip)")
print("="*80)

if len(results_df) > 0:
    print(f"\nTotal chips plotted: {len(results_df)}")
    print(f"Mean TPP/Die Area: {results_df['tpp_per_area'].mean():.2f} Tera-ops/mm²")
    print(f"Median TPP/Die Area: {results_df['tpp_per_area'].median():.2f} Tera-ops/mm²")

    max_idx = results_df['tpp_per_area'].idxmax()
    min_idx = results_df['tpp_per_area'].idxmin()

    print(f"\nMax TPP/Die Area: {results_df.loc[max_idx, 'tpp_per_area']:.2f} Tera-ops/mm²")
    print(f"  Chip: {results_df.loc[max_idx, 'name']}")
    print(f"  Bit length: {results_df.loc[max_idx, 'bit_length']}-bit ({results_df.loc[max_idx, 'perf_type']})")

    print(f"\nMin TPP/Die Area: {results_df.loc[min_idx, 'tpp_per_area']:.2f} Tera-ops/mm²")
    print(f"  Chip: {results_df.loc[min_idx, 'name']}")
    print(f"  Bit length: {results_df.loc[min_idx, 'bit_length']}-bit ({results_df.loc[min_idx, 'perf_type']})")

    # Show distribution by bit length
    print("\nDistribution by Bit Length:")
    for bit_len in sorted(results_df['bit_length'].unique()):
        count = len(results_df[results_df['bit_length'] == bit_len])
        print(f"  {bit_len}-bit: {count} chips")

    # Show state-of-the-art chips
    print("\n" + "="*80)
    print("STATE-OF-THE-ART CHIPS (Highlighted in Green):")
    print("="*80)
    print("Chips that achieved higher TPP/Die Area than any previous chip:\n")
    for idx, row in sota_df.sort_values('year').iterrows():
        print(f"{int(row['year'])}: {row['name']}")
        print(f"  TPP/Die Area: {row['tpp_per_area']:.2f} Tera-ops/mm²")
        print(f"  Bit length: {row['bit_length']}-bit ({row['perf_type']})\n")

    # Show regression statistics
    if len(sota_df) > 1:
        print("="*80)
        print("REGRESSION ANALYSIS (State-of-the-Art Chips):")
        print("="*80)
        log_tpp = np.log10(sota_df['tpp_per_area'])
        years = sota_df['year'].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, log_tpp)

        # Calculate metrics
        annual_multiplier = 10 ** slope
        doubling_time = np.log10(2) / slope if slope > 0 else float('inf')

        print(f"\nR² value: {r_value**2:.4f}")
        print(f"Annual multiplier: {annual_multiplier:.2f}×")
        print(f"Annual growth rate: {(annual_multiplier - 1) * 100:.2f}%")
        print(f"Doubling time: {doubling_time:.2f} years")

# plt.show() #i don't want to show the plot
