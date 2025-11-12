import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Read the timelines data
df = pd.read_csv('data_centers/data_center_timelines.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Get unique datacenters (excluding Google Omaha Nebraska)
datacenters = df['Data center'].unique()
datacenters = [dc for dc in datacenters if dc != 'Google Omaha Nebraska']

# Calculate GW/year and construction cost for each datacenter
results = []

for datacenter in datacenters:
    # Get timeline data
    dc_data = df[df['Data center'] == datacenter].copy()

    if len(dc_data) == 0:
        continue

    # Sort by date
    dc_data = dc_data.sort_values('Date')

    # Find ground clearing date and calculate total time and power
    clearing_date = dc_data['Date'].iloc[0]
    last_date = dc_data['Date'].iloc[-1]

    # Get max power (in GW) and max construction cost
    max_power_gw = dc_data['Power (MW)'].max() / 1000
    max_construction_cost = dc_data['Construction cost (2025 USD billions)'].max()

    # Skip if no construction cost data
    if max_construction_cost == 0 or pd.isna(max_construction_cost):
        continue

    # Calculate construction cost per year
    # Use the time from ground clearing to the last date as the construction period
    days_elapsed = (last_date - clearing_date).days

    if days_elapsed > 0:
        years_elapsed = days_elapsed / 365.25
        construction_cost_per_year = max_construction_cost / years_elapsed

        # Also calculate GW per year using regression (same as workers plot)
        # Find first roof or use clearing date
        roof_indices = dc_data[
            dc_data['Construction status'].str.contains('roof', case=False, na=False)
        ].index

        if len(roof_indices) > 0:
            first_roof_idx = roof_indices[0]
            first_roof_date = dc_data.loc[first_roof_idx, 'Date']
        else:
            first_roof_idx = 0
            first_roof_date = clearing_date

        # Determine the starting point: max of first_roof_date and 2025-01-01
        start_date = max(first_roof_date, pd.Timestamp('2025-01-01'))

        # Calculate days from start for all points
        dc_data['Days from Start'] = (dc_data['Date'] - start_date).dt.days

        # Filter to only include data from start_date onwards
        dc_data_filtered = dc_data[dc_data['Date'] >= start_date].copy()

        if len(dc_data_filtered) >= 2:
            # Calculate GW values
            dc_data_filtered['Power (GW)'] = dc_data_filtered['Power (MW)'] / 1000

            # Calculate regression slope (GW/day)
            dc_x = np.array(dc_data_filtered['Days from Start'])
            dc_y = np.array(dc_data_filtered['Power (GW)'])

            slope, intercept = np.polyfit(dc_x, dc_y, 1)

            # Convert slope from GW/day to GW/year
            gw_per_year = slope * 365.25

            results.append({
                'Data Center': datacenter,
                'Construction Cost per Year ($B/year)': construction_cost_per_year,
                'GW per Year': gw_per_year,
                'Max Power (GW)': max_power_gw,
                'Total Construction Cost ($B)': max_construction_cost,
                'Years Elapsed': years_elapsed
            })

# Create dataframe
results_df = pd.DataFrame(results)

print("Data Centers with construction cost and timeline data:")
print(results_df.to_string(index=False))

# Create scatter plot
fig = go.Figure()

# Add scatter points
fig.add_trace(go.Scatter(
    x=results_df['Construction Cost per Year ($B/year)'],
    y=results_df['GW per Year'],
    mode='markers+text',
    marker=dict(size=12, color='blue'),
    text=results_df['Data Center'],  # Full datacenter name
    textposition='top center',
    textfont=dict(size=7),
    hovertemplate='<b>%{customdata[0]}</b><br>' +
                  'Construction Cost/year: $%{x:.2f}B/year<br>' +
                  'GW/year: %{y:.3f}<br>' +
                  'Total Cost: $%{customdata[1]:.2f}B<br>' +
                  'Max Power: %{customdata[2]:.2f} GW<br>' +
                  '<extra></extra>',
    customdata=results_df[['Data Center', 'Total Construction Cost ($B)', 'Max Power (GW)']].values,
    showlegend=False
))

# Calculate and add regression line (forcing through origin)
if len(results_df) > 1:
    X = results_df['Construction Cost per Year ($B/year)'].values
    y = results_df['GW per Year'].values

    # Fit linear regression through origin (intercept = 0)
    slope = np.sum(X * y) / np.sum(X ** 2)
    intercept = 0

    # Generate regression line from origin
    x_range = np.linspace(0, X.max(), 100)
    y_pred = slope * x_range

    # Calculate R² for regression through origin
    ss_tot = np.sum(y ** 2)
    ss_res = np.sum((y - slope * X) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Add regression line
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name=f'Regression (slope={slope:.4f}, R²={r2:.3f})',
        showlegend=True
    ))

    print(f"\n\nRegression Results (through origin):")
    print(f"Slope: {slope:.6f} (GW/year) per ($B/year) construction spending rate")
    print(f"Intercept: {intercept:.6f} GW/year (forced to 0)")
    print(f"R²: {r2:.4f}")

# Update layout
fig.update_layout(
    title='Datacenter Construction Spending Rate vs Build Rate',
    xaxis_title='Construction Spending Rate (2025 USD billions per year)',
    yaxis_title='Build Rate (GW per Year)',
    width=1000,
    height=700,
    hovermode='closest',
    showlegend=True,
    legend=dict(
        yanchor="bottom",
        y=0.02,
        xanchor="right",
        x=0.98
    )
)

fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)

# Save to PNG
fig.write_image('construction_cost_vs_gw_per_year.png', width=1000, height=700, scale=2)
print("\n\nPlot saved to construction_cost_vs_gw_per_year.png")
