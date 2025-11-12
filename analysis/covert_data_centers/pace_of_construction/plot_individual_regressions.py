import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Read the data
df = pd.read_csv('data_centers/data_center_timelines.csv')
df['Date'] = pd.to_datetime(df['Date'])
today = pd.Timestamp.today()

# Get unique datacenters (excluding Google Omaha Nebraska)
datacenters = df['Data center'].unique()
datacenters = [dc for dc in datacenters if dc != 'Google Omaha Nebraska']

# Define colors
colors = [
    '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3',
    '#FF6692', '#B6E880', '#FF97FF', '#FECB52', '#8B4513', '#2F4F4F'
]

# Calculate number of rows and columns for subplots
n_datacenters = len(datacenters)
n_cols = 3
n_rows = (n_datacenters + n_cols - 1) // n_cols

# Create subplots
fig = make_subplots(
    rows=n_rows,
    cols=n_cols,
    subplot_titles=datacenters,
    vertical_spacing=0.08,
    horizontal_spacing=0.08
)

# For each datacenter, plot actual data and regression line
for idx, datacenter in enumerate(datacenters):
    color = colors[idx % len(colors)]

    # Calculate subplot position
    row = (idx // n_cols) + 1
    col = (idx % n_cols) + 1

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

    # Split into past and future
    past_data = dc_data_filtered[dc_data_filtered['Date'] <= today].copy()
    future_data = dc_data_filtered[dc_data_filtered['Date'] > today].copy()

    # Calculate regression slope (GW/day)
    dc_x = np.array(dc_data_filtered['Days from Start'])
    dc_y = np.array(dc_data_filtered['Power (GW)'])

    if len(dc_x) >= 2:
        slope, intercept = np.polyfit(dc_x, dc_y, 1)
        gw_per_year = slope * 365.25

        # Add actual data points (solid)
        if len(past_data) > 0:
            fig.add_trace(go.Scatter(
                x=past_data['Days from Start'],
                y=past_data['Power (GW)'],
                mode='markers',
                marker=dict(size=6, color=color),
                name=datacenter,
                legendgroup=datacenter,
                showlegend=False,
                hovertemplate='Days: %{x}<br>' +
                              'Power: %{y:.3f} GW<br>' +
                              '<extra></extra>'
            ), row=row, col=col)

        # Add future data points (open circles)
        if len(future_data) > 0:
            fig.add_trace(go.Scatter(
                x=future_data['Days from Start'],
                y=future_data['Power (GW)'],
                mode='markers',
                marker=dict(size=6, color=color, symbol='circle-open'),
                name=datacenter + ' (planned)',
                legendgroup=datacenter,
                showlegend=False,
                hovertemplate='Days: %{x}<br>' +
                              'Power: %{y:.3f} GW (planned)<br>' +
                              '<extra></extra>'
            ), row=row, col=col)

        # Add regression line
        x_range = np.array([dc_x.min(), dc_x.max()])
        y_pred = slope * x_range + intercept

        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_pred,
            mode='lines',
            line=dict(color=color, width=2, dash='dash'),
            name=f'{gw_per_year:.3f} GW/yr',
            legendgroup=datacenter,
            showlegend=False,
            hovertemplate=f'Regression: {gw_per_year:.3f} GW/year<br>' +
                          '<extra></extra>'
        ), row=row, col=col)

# Update layout
fig.update_layout(
    title='Individual Datacenter Build Curves and Regression Lines',
    height=300 * n_rows,
    width=1400,
    hovermode='closest',
    showlegend=False
)

# Update all x and y axes
fig.update_xaxes(showgrid=True, title_text='Days from Start')
fig.update_yaxes(showgrid=True, title_text='Power (GW)')

# Save to PNG
fig.write_image('individual_build_regressions.png', width=1400, height=300 * n_rows, scale=2)
print("Plot saved to individual_build_regressions.png")
