import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import interp1d

# Import the forecast data
from forecasts import p_selfsufficiency_130nm, p_selfsufficiency_28nm, p_selfsufficiency_14nm, p_selfsufficiency_7nm

# Prepare data for plotting
forecasts = {
    '130nm': p_selfsufficiency_130nm,
    '28nm': p_selfsufficiency_28nm,
    '14nm': p_selfsufficiency_14nm,
    '7nm': p_selfsufficiency_7nm
}

# Colors for each node - matching the web app colors
colors = {
    '130nm': '#9B7BB3',  # Purple (web app color 1)
    '28nm': '#5B8DBE',   # Blue (web app color 2)
    '14nm': '#5AA89B',   # Blue-green (web app color 3)
    '7nm': '#F4A261'     # Yellow/Orange (additional color)
}

# Create figure
fig = go.Figure()

# Plot each node's forecast
for node, data in forecasts.items():
    # Extract years and probabilities
    years = np.array([point['year'] for point in data])
    probabilities = np.array([point['probability'] for point in data])

    # Create smooth curve using linear interpolation (only 2 data points)
    years_smooth = np.linspace(years.min(), years.max(), 100)
    interpolator = interp1d(years, probabilities, kind='linear')
    probabilities_smooth = interpolator(years_smooth)

    # Plot the smooth curve
    fig.add_trace(go.Scatter(
        x=years_smooth,
        y=probabilities_smooth,
        mode='lines',
        name=node,
        line=dict(color=colors[node], width=2.0),
        hovertemplate='%{y:.1%}<extra></extra>'
    ))

    # Plot the original data points
    fig.add_trace(go.Scatter(
        x=years,
        y=probabilities,
        mode='markers',
        name=node,
        marker=dict(
            color=colors[node],
            size=10,
            line=dict(color='white', width=1.5)
        ),
        showlegend=False,
        hovertemplate='%{y:.1%}<extra></extra>'
    ))

# Update layout to match web app style
fig.update_layout(
    xaxis=dict(
        title=dict(text='Year', font=dict(size=12)),
        tickfont=dict(size=10),
        range=[2024.5, 2031.5]
    ),
    yaxis=dict(
        title=dict(text='Probability of >90% localization for node', font=dict(size=12)),
        tickfont=dict(size=10),
        tickformat='.0%',
        range=[0, 1]
    ),
    showlegend=True,
    legend=dict(
        x=0.02,
        y=0.98,
        xanchor='left',
        yanchor='top',
        font=dict(size=11),
        bgcolor='rgba(255,255,255,0.95)',
        bordercolor='#ccc',
        borderwidth=1
    ),
    hovermode='closest',
    width=600,
    height=350,
    margin=dict(l=80, r=50, t=50, b=80),
    paper_bgcolor='white',
    plot_bgcolor='white',
    xaxis_gridcolor='rgba(0,0,0,0.1)',
    yaxis_gridcolor='rgba(0,0,0,0.1)',
    xaxis_showgrid=True,
    yaxis_showgrid=True
)

# Save as interactive HTML
fig.write_html('self_sufficiency_forecasts_plotly.html')
print("Interactive Plotly plot saved to self_sufficiency_forecasts_plotly.html")

# Also save as static image (requires kaleido)
try:
    fig.write_image('self_sufficiency_forecasts_plotly.png', width=600, height=350, scale=2)
    print("Static PNG plot saved to self_sufficiency_forecasts_plotly.png")
except Exception as e:
    print(f"Could not save static image (install kaleido if needed): {e}")
