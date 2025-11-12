import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Read the data
df = pd.read_csv('data_centers/data_center_timelines.csv')

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Convert Power from MW to GW
df['Power (GW)'] = df['Power (MW)'] / 1000

# Get unique datacenters
datacenters = df['Data center'].unique()

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
    '#DC143C',  # Crimson
]

# Create figure
fig = go.Figure()

# For each datacenter, find the first "ground clearing" or "land clearing" event
# and plot only data from that point forward
for idx, datacenter in enumerate(datacenters):
    # Assign color based on datacenter index
    color = colors[idx % len(colors)]
    dc_data = df[df['Data center'] == datacenter].copy()

    # Find the first row where construction status mentions "clearing"
    clearing_indices = dc_data[
        dc_data['Construction status'].str.contains('clearing', case=False, na=False)
    ].index

    if len(clearing_indices) > 0:
        # Get the first clearing index
        first_clearing_idx = clearing_indices[0]

        # Filter data from that point forward
        dc_data = dc_data.loc[first_clearing_idx:]

        # Add main line trace for this datacenter
        fig.add_trace(go.Scatter(
            x=dc_data['Date'],
            y=dc_data['Power (GW)'],
            mode='lines+markers',
            name=datacenter,
            line=dict(color=color),
            marker=dict(color=color),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          'Date: %{x|%Y-%m-%d}<br>' +
                          'Power: %{y:.2f} GW<br>' +
                          '<extra></extra>',
            showlegend=True
        ))

        # Add marker for ground clearing (box/square)
        clearing_row = dc_data.iloc[0]
        fig.add_trace(go.Scatter(
            x=[clearing_row['Date']],
            y=[clearing_row['Power (GW)']],
            mode='markers',
            marker=dict(symbol='square', size=10, line=dict(width=2, color='black')),
            name=datacenter,
            legendgroup=datacenter,
            showlegend=False,
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          'Ground Clearing<br>' +
                          'Date: %{x|%Y-%m-%d}<br>' +
                          'Power: %{y:.2f} GW<br>' +
                          '<extra></extra>'
        ))

        # Find the first row where construction status mentions "roof"
        roof_indices = dc_data[
            dc_data['Construction status'].str.contains('roof', case=False, na=False)
        ].index

        if len(roof_indices) > 0:
            first_roof_idx = roof_indices[0]
            roof_row = dc_data.loc[first_roof_idx]

            # Add marker for first roof (triangle)
            fig.add_trace(go.Scatter(
                x=[roof_row['Date']],
                y=[roof_row['Power (GW)']],
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, line=dict(width=2, color='black')),
                name=datacenter,
                legendgroup=datacenter,
                showlegend=False,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'First Roof<br>' +
                              'Date: %{x|%Y-%m-%d}<br>' +
                              'Power: %{y:.2f} GW<br>' +
                              '<extra></extra>'
            ))

# Update layout
fig.update_layout(
    title='Data Center Power Capacity Over Time (from Ground Clearing)',
    xaxis_title='Date',
    yaxis_title='Power (GW)',
    hovermode='closest',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ),
    width=1400,
    height=800
)

# Update axes
fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)

# Save to PNG
fig.write_image('datacenter_power_timeline.png', width=1400, height=800, scale=2)
print("Plot saved to datacenter_power_timeline.png")
