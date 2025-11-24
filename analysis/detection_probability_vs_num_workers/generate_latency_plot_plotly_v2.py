"""
Generate detection latency vs workers plot using Plotly with theme colors
Uses cached Monte Carlo simulation data
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle

# Load the cached plot data from Monte Carlo simulations
with open('model_bayesian_plot_data.pkl', 'rb') as f:
    plot_data = pickle.load(f)

# Extract the cached prediction data
x_pred = plot_data['X_plot_workers']
y_median = plot_data['y_plot']
y_low = plot_data['y_lower']
y_high = plot_data['y_upper']

# Load and prepare the actual data
df = pd.read_csv('nuclear_case_studies.csv')

def extract_years(text):
    """Extract years from 'Function identified' column"""
    if pd.isna(text) or 'NA' in str(text):
        return np.nan
    try:
        import re
        if 'several' in str(text).lower():
            return 6.0
        match = re.search(r'\(([<~]?\d+)\s*yr\)', str(text))
        if match:
            years_str = match.group(1)
            if '<' in years_str:
                return float(years_str.replace('<', '')) / 2
            if '~' in years_str:
                return float(years_str.replace('~', ''))
            return float(years_str)
    except:
        pass
    return np.nan

def extract_workers(text):
    """Extract numeric worker count"""
    if pd.isna(text):
        return np.nan
    try:
        import re
        text_clean = str(text).replace(',', '').replace('~', '')
        match = re.search(r'(\d+)', text_clean)
        if match:
            return float(match.group(1))
    except:
        pass
    return np.nan

df['Detection Latency (years)'] = df['Function identified'].apply(extract_years)
df['Workers (nuclear roles)'] = df['Workers (nuclear roles)'].apply(extract_workers)
df_clean = df.dropna(subset=['Detection Latency (years)', 'Workers (nuclear roles)'])

# Theme colors matching the frontend
colors = {
    'line': '#5B8DBE',
    'fill': 'rgba(91, 141, 190, 0.2)',
    'points': '#5B8DBE'
}

# Create figure
fig = go.Figure()

# Add confidence interval
fig.add_trace(go.Scatter(
    x=x_pred,
    y=y_high,
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo='skip'
))

fig.add_trace(go.Scatter(
    x=x_pred,
    y=y_low,
    mode='lines',
    line=dict(width=0),
    fill='tonexty',
    fillcolor=colors['fill'],
    showlegend=False,
    hoverinfo='skip'
))

# Add median line
fig.add_trace(go.Scatter(
    x=x_pred,
    y=y_median,
    mode='lines',
    line=dict(color=colors['line'], width=2),
    name='Median prediction',
    hovertemplate='workers: %{x}<br>years: %{y:.1f}<extra></extra>'
))

# Add actual data points
fig.add_trace(go.Scatter(
    x=df_clean['Workers (nuclear roles)'],
    y=df_clean['Detection Latency (years)'],
    mode='markers',
    marker=dict(
        size=8,
        color=colors['points'],
        line=dict(color='white', width=1)
    ),
    name='Historical cases',
    text=df_clean['Site'],
    hovertemplate='%{text}<br>workers: %{x}<br>years: %{y:.1f}<extra></extra>',
    showlegend=False
))

# Add text labels for each project
# Sort by x position for better label placement
points_data = [(row['Workers (nuclear roles)'],
                row['Detection Latency (years)'],
                row['Site'])
               for _, row in df_clean.iterrows()]
points_data.sort(key=lambda p: (np.log10(p[0]), p[1]))

# Simple label placement with slight offsets to avoid overlap
for i, (x_pos, y_pos, site_name) in enumerate(points_data):
    # Shorten some long names for better display
    display_name = site_name
    if 'Iranian Fordow' in site_name:
        display_name = 'Fordow'
    elif 'Syrian Al-Kibar' in site_name:
        display_name = 'Al-Kibar'
    elif 'North Korean 2010' in site_name:
        display_name = 'NK 2010'
    elif 'Saudi enrichment' in site_name:
        display_name = 'Saudi'
    elif 'Iranian Natanz' in site_name:
        display_name = 'Natanz'
    elif 'Libya centrifuge' in site_name:
        display_name = 'Libya'
    elif 'Pakistan Kahuta' in site_name:
        display_name = 'Kahuta'
    elif 'North Korea 1980s' in site_name:
        display_name = 'NK 1980s'
    elif 'Iraq PC-3' in site_name:
        display_name = 'Iraq PC-3'
    elif 'South Africa' in site_name:
        display_name = 'S. Africa'
    elif 'Iran – Lavizan' in site_name:
        display_name = 'Lavizan'
    elif 'Israel – Dimona' in site_name:
        display_name = 'Dimona'

    # Add text annotation with line connector
    fig.add_annotation(
        x=x_pos,
        y=y_pos,
        text=display_name,
        xanchor='left',
        yanchor='middle',
        xshift=10,
        font=dict(size=9, color='#333'),
        showarrow=True,
        arrowhead=0,
        arrowsize=0.5,
        arrowwidth=0.5,
        arrowcolor='rgba(0,0,0,0.3)',
        ax=20,
        ay=0
    )

# Update layout
fig.update_layout(
    xaxis=dict(
        title=dict(text='Nuclear-role workers', font=dict(size=13)),
        type='log',
        tickfont=dict(size=10),
        tickvals=[100, 1000, 10000],
        ticktext=['100', '1,000', '10,000'],
        gridcolor='rgba(128, 128, 128, 0.2)'
    ),
    yaxis=dict(
        title=dict(text='Detection latency (years)', font=dict(size=13)),
        tickfont=dict(size=10),
        rangemode='tozero',
        gridcolor='rgba(128, 128, 128, 0.2)'
    ),
    showlegend=True,
    legend=dict(
        x=0.02,
        y=0.98,
        xanchor='left',
        yanchor='top',
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='#ccc',
        borderwidth=1,
        font=dict(size=10)
    ),
    hovermode='closest',
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(l=60, r=150, t=20, b=60),
    height=400,
    width=900
)

# Save as static image with new filename
fig.write_image('detection_latency_vs_workers_plotly.png', width=900, height=400, scale=2)
print("Plot saved as 'detection_latency_vs_workers_plotly.png'")

# Also save as HTML for reference
fig.write_html('detection_latency_vs_workers_plotly.html')
print("Interactive plot saved as 'detection_latency_vs_workers_plotly.html'")
