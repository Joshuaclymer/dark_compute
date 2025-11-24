"""
Generate detection latency vs workers plot using Plotly (bayesian style with labels)
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

df['detection_years'] = df['Function identified'].apply(extract_years)
df['num_workers'] = df['Workers (nuclear roles)'].apply(extract_workers)
df_clean = df.dropna(subset=['detection_years', 'num_workers'])

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
    hoverinfo='skip',
    name=''
))

fig.add_trace(go.Scatter(
    x=x_pred,
    y=y_low,
    mode='lines',
    line=dict(width=0),
    fill='tonexty',
    fillcolor=colors['fill'],
    name='90% Confidence Interval',
    hoverinfo='skip'
))

# Add median line
fig.add_trace(go.Scatter(
    x=x_pred,
    y=y_median,
    mode='lines',
    line=dict(color=colors['line'], width=2),
    name='Posterior Mean',
    hovertemplate='workers: %{x}<br>years: %{y:.1f}<extra></extra>'
))

# Add actual data points (no legend)
fig.add_trace(go.Scatter(
    x=df_clean['num_workers'],
    y=df_clean['detection_years'],
    mode='markers',
    marker=dict(
        size=8,
        color=colors['points'],
        line=dict(color='white', width=1)
    ),
    name='',
    text=df_clean['Site'],
    hovertemplate='%{text}<br>workers: %{x}<br>years: %{y:.1f}<extra></extra>',
    showlegend=False
))

# Add text labels for each project using smart positioning algorithm
# Sort by x position for better label placement
points_data = [(row['num_workers'], row['detection_years'], row['Site'])
               for _, row in df_clean.iterrows()]
points_data.sort(key=lambda p: (np.log10(p[0]), p[1]))

# Smart label positioning to avoid overlaps
label_positions = []

for x_pos, y_pos, site_name in points_data:
    offset_y = y_pos

    max_iterations = 100
    for iteration in range(max_iterations):
        conflict = False
        for prev_x, prev_y in label_positions:
            x_log_dist = abs(np.log10(x_pos) - np.log10(prev_x))
            y_dist = abs(offset_y - prev_y)

            if x_log_dist < 0.6 and y_dist < 1.8:
                conflict = True
                offset_y += 0.9
                break

        if not conflict:
            break

    label_positions.append((x_pos, offset_y))

    # Add line connector if label was moved significantly
    if abs(offset_y - y_pos) > 0.4:
        fig.add_trace(go.Scatter(
            x=[x_pos, x_pos * 1.15],
            y=[y_pos, offset_y],
            mode='lines',
            line=dict(color='rgba(0,0,0,0.4)', width=0.5),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add text label as a scatter trace (not annotation) for better rendering
    x_offset = x_pos * 1.18
    fig.add_trace(go.Scatter(
        x=[x_offset],
        y=[offset_y],
        mode='text',
        text=[site_name],
        textposition='middle right',
        textfont=dict(size=9, color='#333'),
        showlegend=False,
        hoverinfo='skip'
    ))

# Update layout
fig.update_layout(
    xaxis=dict(
        title=dict(text='Nuclear-role workers', font=dict(size=13)),
        type='log',
        tickfont=dict(size=10),
        tickvals=[100, 1000, 10000],
        ticktext=['100', '1,000', '10,000'],
        gridcolor='rgba(128, 128, 128, 0.2)',
        range=[np.log10(20), np.log10(15000)]  # Extend range to accommodate labels
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
    margin=dict(l=70, r=150, t=20, b=60),
    height=450,
    width=1100
)

# Save as static image with new filename (don't override bayesian.png)
fig.write_image('detection_latency_vs_workers_bayesian_plotly.png', width=1100, height=450, scale=2)
print("Plot saved as 'detection_latency_vs_workers_bayesian_plotly.png'")

# Also save as HTML for reference
fig.write_html('detection_latency_vs_workers_bayesian_plotly.html')
print("Interactive plot saved as 'detection_latency_vs_workers_bayesian_plotly.html'")
