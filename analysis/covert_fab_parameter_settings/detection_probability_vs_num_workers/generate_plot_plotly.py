"""
Generate a Plotly version of detection_probability_vs_time plot
using the same plotting style as the web app.

This script uses cached data from model_bayesian_plot_data.pkl
"""

import pickle
import plotly.graph_objects as go
import numpy as np

# Load cached plot data
print("Loading cached plot data...")
with open('model_bayesian_plot_data.pkl', 'rb') as f:
    plot_data = pickle.load(f)

years = plot_data['years']
worker_counts = plot_data['worker_counts']
p_detected_dict = plot_data['p_detected_dict']

print(f"Data loaded. Generating plot...")

# Create figure
fig = go.Figure()

# Colors and labels - matching the web app style
colors = ['#9B7BB3', '#5B8DBE', '#5AA89B']  # Purple, Blue, Blue-green (from web app)
labels = ['100 high-context workers involved', '1,000 high-context workers involved', '10,000 high-context workers involved']

# Plot each worker count
for workers, color, label in zip(worker_counts, colors, labels):
    p_detected = p_detected_dict[workers]

    fig.add_trace(go.Scatter(
        x=years,
        y=p_detected,
        mode='lines',
        name=label,
        line=dict(color=color, width=2.0),
        hovertemplate='%{y:.3f}<extra></extra>'
    ))

# Update layout to match web app style
fig.update_layout(
    xaxis=dict(
        title=dict(text='Years after breaking ground', font=dict(size=10)),
        tickfont=dict(size=10),
        range=[0, 12]
    ),
    yaxis=dict(
        title=dict(text='P(strong evidence of covert project)', font=dict(size=10)),
        tickfont=dict(size=10),
        range=[0, 1]
    ),
    showlegend=True,
    legend=dict(
        x=0.98,
        y=0.02,
        xanchor='right',
        yanchor='bottom',
        font=dict(size=9),
        bgcolor='rgba(255,255,255,0)',
        bordercolor='rgba(0,0,0,0)',
        borderwidth=0
    ),
    hovermode='closest',
    width=600,
    height=350,
    margin=dict(l=80, r=50, t=30, b=80),
    paper_bgcolor='white',
    plot_bgcolor='white',
    xaxis_gridcolor='rgba(0,0,0,0.1)',
    yaxis_gridcolor='rgba(0,0,0,0.1)',
    xaxis_showgrid=True,
    yaxis_showgrid=True
)

# Save as interactive HTML
fig.write_html('detection_probability_vs_time_plotly.html')
print("Interactive Plotly plot saved to detection_probability_vs_time_plotly.html")

# Also save as static image (requires kaleido)
try:
    fig.write_image('detection_probability_vs_time_plotly.png', width=600, height=350, scale=2)
    print("Static PNG plot saved to detection_probability_vs_time_plotly.png")
except Exception as e:
    print(f"Could not save static image (install kaleido if needed): {e}")
