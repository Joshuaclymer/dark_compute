import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Read the data
df = pd.read_csv('data_centers/data_center_timelines.csv')

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Convert Power from MW to GW
df['Power (GW)'] = df['Power (MW)'] / 1000

# Get current date
today = pd.Timestamp.today()

# Get unique datacenters (excluding Google Omaha Nebraska)
datacenters = df['Data center'].unique()
datacenters = [dc for dc in datacenters if dc != 'Google Omaha Nebraska']

# Create figure with subplots
from plotly.subplots import make_subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Power Capacity Over Time', 'Estimated Construction Workers Over Time'),
    horizontal_spacing=0.12,
    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
)

# Define a color palette with enough unique colors for all datacenters
colors = [
    '#636EFA',  # Blue
    '#EF553B',  # Red
    '#00CC96',  # Teal
    '#AB63FA',  # Purple
    '#FFA15A',  # Orange
    '#19D3F3',  # Cyan
    '#FF6692',  # Pink
    '#B6E880',  # Light green
    '#FF97FF',  # Light purple
    '#FECB52',  # Yellow
    '#8B4513',  # Brown
    '#2F4F4F',  # Dark slate gray
]

# Collect individual regression lines for each datacenter
individual_slopes = []
individual_intercepts = []
all_x_range = []

# For each datacenter, find the first "roof" event and normalize time from that point
for idx, datacenter in enumerate(datacenters):
    # Assign color based on datacenter index
    color = colors[idx % len(colors)]
    dc_data = df[df['Data center'] == datacenter].copy()

    # Find the first row where construction status mentions "clearing"
    clearing_indices = dc_data[
        dc_data['Construction status'].str.contains('clearing', case=False, na=False)
    ].index

    if len(clearing_indices) == 0:
        continue

    # Get the first clearing index
    first_clearing_idx = clearing_indices[0]

    # Filter data from clearing point forward
    dc_data = dc_data.loc[first_clearing_idx:].copy()

    # Find the first row where construction status mentions "roof"
    roof_indices = dc_data[
        dc_data['Construction status'].str.contains('roof', case=False, na=False)
    ].index

    # Use first roof date if available, otherwise use clearing date
    if len(roof_indices) > 0:
        first_roof_idx = roof_indices[0]
        first_roof_date = dc_data.loc[first_roof_idx, 'Date']
    else:
        # No roof data, use clearing date as the reference point
        first_roof_idx = first_clearing_idx
        first_roof_date = dc_data.loc[first_clearing_idx, 'Date']

    # Determine the starting point: max of first_roof_date and 2025-01-01
    start_date = max(first_roof_date, pd.Timestamp('2025-01-01'))

    # Calculate days from the starting point for all points
    dc_data['Days from Start'] = (dc_data['Date'] - start_date).dt.days

    # If start_date is after first_roof_date, we need to add a point at start_date
    if start_date > first_roof_date:
        # Find the last data point before or at start_date
        before_start = dc_data[dc_data['Date'] <= start_date]
        if len(before_start) > 0:
            # Get the last value before start_date
            last_before = before_start.iloc[-1].copy()

            # Create a new row at start_date with the same power value
            start_row = last_before.copy()
            start_row['Date'] = start_date
            start_row['Days from Start'] = 0

            # Add this row to dc_data
            dc_data = pd.concat([pd.DataFrame([start_row]), dc_data]).sort_values('Date').reset_index(drop=True)

    # Filter to only include data from start_date onwards
    dc_data = dc_data[dc_data['Date'] >= start_date].copy()

    if len(dc_data) == 0:
        continue

    # Split data into past and future
    past_data = dc_data[dc_data['Date'] <= today].copy()
    future_data = dc_data[dc_data['Date'] > today].copy()

    # Calculate individual regression line for this datacenter
    if len(dc_data) >= 2:  # Need at least 2 points for regression
        dc_x = np.array(dc_data['Days from Start'])
        dc_y = np.array(dc_data['Power (GW)'])

        # Fit linear regression for this datacenter
        dc_slope, dc_intercept = np.polyfit(dc_x, dc_y, 1)

        individual_slopes.append(dc_slope)
        individual_intercepts.append(dc_intercept)
        all_x_range.extend(dc_x.tolist())

    # Calculate workers based on construction cost rate
    # Assumptions: $75k/year per worker, labor is 75% of construction costs
    worker_cost_per_day = 75000 / 365  # ~$205 per day per worker

    # Sort by date
    dc_data_sorted = dc_data.sort_values('Date').copy()

    # Find ground clearing date for this datacenter
    clearing_date = dc_data_sorted['Date'].iloc[0]

    # Calculate days from clearing for each point
    dc_data_sorted['Days from Clearing'] = (dc_data_sorted['Date'] - clearing_date).dt.days

    # Calculate average daily spending rate and workers
    # For each point, the construction cost represents total spending from clearing to that date
    # Average spending per day = Total cost / Days elapsed
    # Workers = (Avg spending per day * labor fraction) / worker cost per day

    dc_data_sorted['Estimated workers'] = 0.0

    for idx in dc_data_sorted.index:
        days_elapsed = dc_data_sorted.loc[idx, 'Days from Clearing']
        construction_cost = dc_data_sorted.loc[idx, 'Construction cost (2025 USD billions)']

        if days_elapsed > 0 and construction_cost > 0:
            # Total cost / days = average $/day
            avg_spending_per_day = (construction_cost * 1e9) / days_elapsed
            # Workers = (spending per day * 75% labor) / (worker cost per day)
            workers = (avg_spending_per_day * 0.75) / worker_cost_per_day
            dc_data_sorted.loc[idx, 'Estimated workers'] = workers

    # Add solid line for past/current data (power subplot)
    if len(past_data) > 0:
        fig.add_trace(go.Scatter(
                x=past_data['Days from Start'],
                y=past_data['Power (GW)'],
                mode='lines+markers',
                name=datacenter,
                line=dict(dash='solid', color=color),
                marker=dict(color=color),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'Days from Start: %{x}<br>' +
                              'Power: %{y:.2f} GW<br>' +
                              '<extra></extra>',
                showlegend=True,
                legendgroup=datacenter
        ), row=1, col=1)

        # Add workers subplot
        past_data_workers = dc_data_sorted[dc_data_sorted['Date'] <= today].copy()
        if len(past_data_workers) > 0:
            fig.add_trace(go.Scatter(
                x=past_data_workers['Days from Start'],
                y=past_data_workers['Estimated workers'],
                mode='lines+markers',
                name=datacenter,
                line=dict(dash='solid', color=color),
                marker=dict(color=color),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'Days from Start: %{x}<br>' +
                              'Workers: %{y:.0f}<br>' +
                              '<extra></extra>',
                showlegend=False,
                legendgroup=datacenter
            ), row=1, col=2)

        # Add dotted line for future/planned data (power)
        if len(future_data) > 0:
            # Connect the last past point to future data
            if len(past_data) > 0:
                last_past = past_data.iloc[-1]
                future_data = pd.concat([pd.DataFrame([last_past]), future_data])

            fig.add_trace(go.Scatter(
                x=future_data['Days from Start'],
                y=future_data['Power (GW)'],
                mode='lines+markers',
                name=datacenter + ' (planned)',
                line=dict(dash='dot', color=color),
                marker=dict(symbol='circle-open', color=color),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'Days from Start: %{x}<br>' +
                              'Power: %{y:.2f} GW (planned)<br>' +
                              '<extra></extra>',
                showlegend=False,
                legendgroup=datacenter
            ), row=1, col=1)

            # Add dotted line for future workers
            future_data_workers = dc_data_sorted[dc_data_sorted['Date'] > today].copy()
            if len(future_data_workers) > 0:
                if len(past_data_workers) > 0:
                    last_past_worker = past_data_workers.iloc[-1]
                    future_data_workers = pd.concat([pd.DataFrame([last_past_worker]), future_data_workers])

                fig.add_trace(go.Scatter(
                    x=future_data_workers['Days from Start'],
                    y=future_data_workers['Estimated workers'],
                    mode='lines+markers',
                    name=datacenter + ' (planned)',
                    line=dict(dash='dot', color=color),
                    marker=dict(symbol='circle-open', color=color),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  'Days from Start: %{x}<br>' +
                                  'Workers: %{y:.0f} (planned)<br>' +
                                  '<extra></extra>',
                    showlegend=False,
                    legendgroup=datacenter
                ), row=1, col=2)

        # Add marker for first roof (triangle) at x=0 if first roof is the start point
        if first_roof_date >= pd.Timestamp('2025-01-01'):
            first_point = dc_data.iloc[0]
            fig.add_trace(go.Scatter(
                x=[0],
                y=[first_point['Power (GW)']],
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, line=dict(width=2, color='black')),
                name=datacenter,
                legendgroup=datacenter,
                showlegend=False,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'First Roof<br>' +
                              'Days from Start: %{x}<br>' +
                              'Power: %{y:.2f} GW<br>' +
                              '<extra></extra>'
            ), row=1, col=1)

# Calculate average regression line from individual datacenter regressions
if len(individual_slopes) > 0:
    # Average the slopes and intercepts (equal weight per datacenter)
    avg_slope = np.mean(individual_slopes)
    avg_intercept = np.mean(individual_intercepts)

    # Generate points for the regression line
    x_range = np.linspace(min(all_x_range), max(all_x_range), 100)
    y_pred = avg_slope * x_range + avg_intercept

    # Add regression line to plot (solid and thicker)
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        name=f'Avg Regression (slope={avg_slope:.4f} GW/day)',
        line=dict(color='black', width=4, dash='solid'),
        showlegend=True
    ), row=1, col=1)

    print(f"\nRegression Results (averaged across {len(individual_slopes)} datacenters):")
    print(f"Average Slope: {avg_slope:.6f} GW/day")
    print(f"Average Intercept: {avg_intercept:.6f} GW")
    print(f"\nIndividual datacenter slopes:")
    for i, (dc, slope) in enumerate(zip(datacenters, individual_slopes)):
        print(f"  {dc}: {slope:.6f} GW/day")

# Print all datacenter names
print("\n\nDatacenter Names:")
print("=" * 60)
for datacenter in datacenters:
    print(datacenter)
print("=" * 60)

# Update layout
fig.update_layout(
    title='Data Center Power Capacity and Construction Workers Over Time',
    hovermode='closest',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ),
    width=2000,
    height=800
)

# Update axes
fig.update_xaxes(title_text='Days from Start (First Roof or 2025, whichever is later)', showgrid=True, row=1, col=1)
fig.update_xaxes(title_text='Days from Start (First Roof or 2025, whichever is later)', showgrid=True, row=1, col=2)
fig.update_yaxes(title_text='Power (GW)', showgrid=True, row=1, col=1)
fig.update_yaxes(title_text='Estimated Workers', showgrid=True, row=1, col=2)

# Save to PNG
fig.write_image('datacenter_power_timeline_normalized.png', width=2000, height=800, scale=2)
print("Plot saved to datacenter_power_timeline_normalized.png")
