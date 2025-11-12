import pandas as pd
import numpy as np

# Read the data
df = pd.read_csv('data_centers/data_center_timelines.csv')
df['Date'] = pd.to_datetime(df['Date'])
today = pd.Timestamp.today()

# Get unique datacenters (excluding Google Omaha Nebraska)
datacenters = df['Data center'].unique()
datacenters = [dc for dc in datacenters if dc != 'Google Omaha Nebraska']

# Calculate GW/year for each datacenter
gw_per_year_results = {}

for datacenter in datacenters:
    dc_data = df[df['Data center'] == datacenter].copy()

    if len(dc_data) == 0:
        continue

    dc_data = dc_data.sort_values('Date')

    # Find ground clearing
    clearing_indices = dc_data[
        dc_data['Construction status'].str.contains('clearing', case=False, na=False)
    ].index

    if len(clearing_indices) == 0:
        continue

    first_clearing_idx = clearing_indices[0]
    dc_data = dc_data.loc[first_clearing_idx:].copy()

    # Find first roof or use clearing date
    roof_indices = dc_data[
        dc_data['Construction status'].str.contains('roof', case=False, na=False)
    ].index

    if len(roof_indices) > 0:
        first_roof_idx = roof_indices[0]
        first_roof_date = dc_data.loc[first_roof_idx, 'Date']
    else:
        first_roof_idx = first_clearing_idx
        first_roof_date = dc_data.loc[first_clearing_idx, 'Date']

    # Determine the starting point: max of first_roof_date and 2025-01-01
    start_date = max(first_roof_date, pd.Timestamp('2025-01-01'))

    # Filter to only include data from start_date onwards (after first roof AND after Jan 1, 2025)
    # AND within 1 year of today
    one_year_from_today = today + pd.DateOffset(years=1)
    dc_data_filtered = dc_data[
        (dc_data['Date'] >= start_date) &
        (dc_data['Date'] <= one_year_from_today)
    ].copy()

    # Calculate days from start for filtered points only
    dc_data_filtered['Days from Start'] = (dc_data_filtered['Date'] - start_date).dt.days

    if len(dc_data_filtered) < 2:
        continue

    # Calculate GW values
    dc_data_filtered['Power (GW)'] = dc_data_filtered['Power (MW)'] / 1000

    # Calculate regression slope (GW/day)
    dc_x = np.array(dc_data_filtered['Days from Start'])
    dc_y = np.array(dc_data_filtered['Power (GW)'])

    if len(dc_x) >= 2:
        slope, intercept = np.polyfit(dc_x, dc_y, 1)
        gw_per_year = slope * 365.25
        gw_per_year_results[datacenter] = gw_per_year

# Print results
print("Updated GW/year values:")
print("=" * 80)
for datacenter, gw_per_year in sorted(gw_per_year_results.items()):
    print(f"{datacenter:60s} {gw_per_year:.3f}")
