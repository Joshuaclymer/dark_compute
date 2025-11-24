"""
Generate detection latency vs workers plot using Plotly with theme colors
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle

# Load the saved Bayesian model data
with open('model_bayesian_data.pkl', 'rb') as f:
    saved_data = pickle.load(f)

posterior_samples = saved_data['posterior_samples']

# Calculate posterior means
A_mean = np.mean(posterior_samples[:, 0])
B_mean = np.mean(posterior_samples[:, 1])

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

# Generate prediction curve
x_pred = np.logspace(np.log10(50), np.log10(5000), 100)

# Generate predictions from posterior samples
n_pred_samples = 500
pred_samples = []

for i in np.random.choice(len(posterior_samples), n_pred_samples, replace=False):
    A, B, sigma_sq = posterior_samples[i]
    y_pred = A / np.log10(x_pred)**B
    pred_samples.append(y_pred)

pred_samples = np.array(pred_samples)
y_median = np.median(pred_samples, axis=0)
y_low = np.percentile(pred_samples, 25, axis=0)
y_high = np.percentile(pred_samples, 75, axis=0)

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
    hovertemplate='%{text}<br>workers: %{x}<br>years: %{y:.1f}<extra></extra>'
))

# Update layout
fig.update_layout(
    xaxis=dict(
        title=dict(text='nuclear-role workers', font=dict(size=13)),
        type='log',
        tickfont=dict(size=10),
        tickvals=[100, 1000, 10000],
        ticktext=['100', '1,000', '10,000'],
        gridcolor='rgba(128, 128, 128, 0.2)'
    ),
    yaxis=dict(
        title=dict(text='detection latency (years)', font=dict(size=13)),
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
    margin=dict(l=60, r=20, t=20, b=60),
    height=350,
    width=600
)

# Save as static image
fig.write_image('detection_latency_vs_workers_bayesian.png', width=600, height=350, scale=2)
print("Plot saved as 'detection_latency_vs_workers_bayesian.png'")

# Also save as HTML for reference
fig.write_html('detection_latency_vs_workers_bayesian.html')
print("Interactive plot saved as 'detection_latency_vs_workers_bayesian.html'")
