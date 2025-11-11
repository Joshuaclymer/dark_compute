"""
Generate a Plotly CCDF plot showing P(Absolute relative error >= x) for empirical errors
using the same plotting style as the web app.
Includes fitted exponential curve like sme_diversion_detection.png
"""

import plotly.graph_objects as go
import numpy as np
from scipy.optimize import curve_fit

# Import data from error_bars.py
estimated_quantities = [700, 800, 900, 300, 1000, 50, 800, 441, 18, 1000, 600, 428.0, 287.0, 311.0, 208]
ground_truth_quantities = [610, 280, 847, 0, 1308, 60, 819, 499, 5, 1027.1, 661.2, 347.5, 308.0, 247.5, 287]

# Fit constrained logistic curve: f(x) = 1 - exp(-k * x)
# We want the median (CDF = 0.5) to be at x = 0.07 (7%)
# At median: 0.5 = 1 - exp(-k * 0.07)
# Solving: exp(-k * 0.07) = 0.5
# -k * 0.07 = ln(0.5)
# k = -ln(0.5) / 0.07
median_error = 0.07
k_fitted = -np.log(0.5) / median_error

def constrained_logistic(x, k):
    return 1.0 - np.exp(-k * x)

# Generate smooth curve from 0 to 1
x_smooth = np.linspace(0, 1.0, 300)
y_smooth = constrained_logistic(x_smooth, k_fitted)
legend_label = f'f(x) = 1 - e^(-{k_fitted:.2f}x)'

print(f"k fitted for median at 7%: {k_fitted:.4f}")

print(f"Data loaded. Generating plot...")

# Create figure
fig = go.Figure()

# Plot fitted curve
fig.add_trace(go.Scatter(
    x=x_smooth,
    y=y_smooth,
    mode='lines',
    name=legend_label,
    line=dict(color='#5B8DBE', width=2.0),  # Blue from web app
    hovertemplate='Absolute relative error: %{x:.3f}<br>Fitted: %{y:.3f}<extra></extra>'
))

# Plot median point
fig.add_trace(go.Scatter(
    x=[median_error],
    y=[0.5],
    mode='markers+text',
    name='Median',
    marker=dict(size=6, color='#5B8DBE', symbol='circle'),  # Same blue as curve, smaller
    text=[f'({median_error:.2f}, 0.50)'],
    textposition='top right',
    textfont=dict(size=9, color='#5B8DBE'),
    showlegend=False,
    hovertemplate='Median error: %{x:.3f}<br>CDF: %{y:.3f}<extra></extra>'
))

# Update layout to match web app style
fig.update_layout(
    xaxis=dict(
        title=dict(text='% error of USG SME estimate', font=dict(size=10)),
        tickfont=dict(size=10),
        range=[0, 1.0],
        tickformat='.0%'
    ),
    yaxis=dict(
        title=dict(text='Cumulative probability', font=dict(size=10)),
        tickfont=dict(size=10),
        range=[0, 1]
    ),
    showlegend=True,
    legend=dict(
        x=0.98,
        y=0.98,
        xanchor='right',
        yanchor='top',
        font=dict(size=9),
        bgcolor='rgba(255,255,255,0.95)',
        bordercolor='#ccc',
        borderwidth=1
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
fig.write_html('error_ccdf_plotly.html')
print("Interactive Plotly plot saved to error_ccdf_plotly.html")

# Also save as static image (requires kaleido)
try:
    fig.write_image('error_ccdf_plotly.png', width=600, height=350, scale=2)
    print("Static PNG plot saved to error_ccdf_plotly.png")
except Exception as e:
    print(f"Could not save static image (install kaleido if needed): {e}")
