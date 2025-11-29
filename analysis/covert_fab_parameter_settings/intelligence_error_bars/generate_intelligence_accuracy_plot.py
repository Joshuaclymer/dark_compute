"""
Generate intelligence accuracy plot for tooltips.
This replicates the intelligenceAccuracyPlot from dark_compute_detection_section.js
Shows estimate vs ground truth with median error margin.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Color scheme matching frontend
BLUE = '#5B8DBE'
POINT_COLOR = '#5B9DB5'

# Data for stated error bars
stated_error_bars = [
    {"category": "Nuclear Warheads", "min": 150, "max": 160, "date": "1984"},
    {"category": "Nuclear Warheads", "min": 140, "max": 157, "date": "1999"},
    {"category": "Nuclear Warheads", "min": 225, "max": 300, "date": "1984"},
    {"category": "Nuclear Warheads", "min": 60, "max": 80, "date": "1999"},
    {"category": "Fissile material (kg)", "min": 25, "max": 35, "date": "1994"},
    {"category": "Fissile material (kg)", "min": 30, "max": 50, "date": "2007"},
    {"category": "Fissile material (kg)", "min": 17, "max": 33, "date": "1994"},
    {"category": "Fissile material (kg)", "min": 335, "max": 400, "date": "1998"},
    {"category": "Fissile material (kg)", "min": 330, "max": 580, "date": "1996"},
    {"category": "Fissile material (kg)", "min": 240, "max": 395, "date": "2000"},
    {"category": "ICBM launchers", "min": 10, "max": 25, "date": "1961"},
    {"category": "ICBM launchers", "min": 10, "max": 25, "date": "1961"},
    {"category": "ICBM launchers", "min": 105, "max": 120, "date": "1963"},
    {"category": "ICBM launchers", "min": 200, "max": 240, "date": "1964"},
    {"category": "Intercontinental missiles", "min": 180, "max": 190, "date": "2019"},
    {"category": "Intercontinental missiles", "min": 200, "max": 300, "date": "2025"},
    {"category": "Intercontinental missiles", "min": 192, "max": 192, "date": "2024"}
]

# Calculate central estimates and bounds
central_estimates = []
lower_bounds = []
upper_bounds = []
upper_percent_errors = []
lower_percent_errors = []

for entry in stated_error_bars:
    central = (entry["min"] + entry["max"]) / 2
    central_estimates.append(central)
    lower_bounds.append(entry["min"])
    upper_bounds.append(entry["max"])

    upper_error = ((entry["max"] - central) / central) * 100
    lower_error = ((central - entry["min"]) / central) * 100
    upper_percent_errors.append(upper_error)
    lower_percent_errors.append(lower_error)

median_upper_error = np.median(upper_percent_errors)
median_lower_error = np.median(lower_percent_errors)
upper_slope = 1 + (median_upper_error / 100)
lower_slope = 1 - (median_lower_error / 100)

# Data for estimate vs reality
estimates = [700, 800, 900, 300, 1000, 50, 800, 441, 18, 1000, 600, 428.0, 287.0, 311.0, 208]
ground_truths = [610, 280, 847, 0, 1308, 60, 819, 499, 5, 1027.1, 661.2, 347.5, 308.0, 247.5, 287]

# Calculate median estimate error
estimate_percent_errors = []
for i in range(len(estimates)):
    if ground_truths[i] != 0:
        estimate_percent_errors.append(abs((estimates[i] - ground_truths[i]) / ground_truths[i]) * 100)

median_estimate_error = np.median(estimate_percent_errors)
estimate_upper_slope = 1 + (median_estimate_error / 100)
estimate_lower_slope = 1 - (median_estimate_error / 100)

# Create figure with subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=('Stated estimate ranges', 'Estimate vs. ground truth'),
                    horizontal_spacing=0.12)

# Left subplot: Stated ranges
max_range = max(max(central_estimates), max(upper_bounds))
x_line = np.linspace(0, max_range, 100)

# Add error bar lines
for i in range(len(central_estimates)):
    fig.add_trace(go.Scatter(
        x=[central_estimates[i], central_estimates[i]],
        y=[lower_bounds[i], upper_bounds[i]],
        mode='lines',
        line=dict(color=POINT_COLOR, width=1),
        opacity=0.3,
        showlegend=False,
        hoverinfo='skip'
    ), row=1, col=1)

# Add upper bound points
fig.add_trace(go.Scatter(
    x=central_estimates,
    y=upper_bounds,
    mode='markers',
    marker=dict(color=POINT_COLOR, size=6),
    showlegend=False,
    hoverinfo='skip'
), row=1, col=1)

# Add lower bound points
fig.add_trace(go.Scatter(
    x=central_estimates,
    y=lower_bounds,
    mode='markers',
    marker=dict(color=POINT_COLOR, size=6),
    showlegend=False,
    hoverinfo='skip'
), row=1, col=1)

# Add median error region for left subplot
fig.add_trace(go.Scatter(
    x=np.concatenate([x_line, x_line[::-1]]),
    y=np.concatenate([lower_slope * x_line, upper_slope * x_line[::-1]]),
    fill='toself',
    fillcolor='rgba(200, 200, 200, 0.3)',
    line=dict(width=0),
    name=f'Median error = {median_upper_error:.1f}%',
    showlegend=True,
    hoverinfo='skip',
    legendgroup='left'
), row=1, col=1)

# Add y=x line for left subplot
fig.add_trace(go.Scatter(
    x=x_line,
    y=x_line,
    mode='lines',
    line=dict(color='grey', width=1),
    opacity=0.5,
    showlegend=False,
    hoverinfo='skip'
), row=1, col=1)

# Right subplot: Estimate vs reality
max_range_est = max(max(estimates), max(ground_truths))
x_line_est = np.linspace(0, max_range_est, 100)

# Add estimate points
fig.add_trace(go.Scatter(
    x=ground_truths,
    y=estimates,
    mode='markers',
    marker=dict(color=POINT_COLOR, size=6),
    showlegend=False,
    hoverinfo='skip'
), row=1, col=2)

# Add median error region for right subplot
fig.add_trace(go.Scatter(
    x=np.concatenate([x_line_est, x_line_est[::-1]]),
    y=np.concatenate([estimate_lower_slope * x_line_est, estimate_upper_slope * x_line_est[::-1]]),
    fill='toself',
    fillcolor='rgba(200, 200, 200, 0.3)',
    line=dict(width=0),
    name=f'Median error = {median_estimate_error:.1f}%',
    showlegend=True,
    hoverinfo='skip',
    legendgroup='right'
), row=1, col=2)

# Add y=x line for right subplot
fig.add_trace(go.Scatter(
    x=x_line_est,
    y=x_line_est,
    mode='lines',
    line=dict(color='grey', width=1),
    opacity=0.5,
    showlegend=False,
    hoverinfo='skip'
), row=1, col=2)

# Update layout
fig.update_layout(
    width=700,
    height=350,
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(size=10),
    margin=dict(l=50, r=20, t=40, b=60),
    showlegend=True,
    legend=dict(
        x=0.02, y=0.98,
        xanchor='left', yanchor='top',
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='#ccc', borderwidth=1,
        font=dict(size=8)
    )
)

fig.update_xaxes(title_text='Central Estimate', gridcolor='rgba(128,128,128,0.2)',
                 linecolor='#ccc', showline=True, row=1, col=1)
fig.update_yaxes(title_text='Stated Range', gridcolor='rgba(128,128,128,0.2)',
                 linecolor='#ccc', showline=True, row=1, col=1)
fig.update_xaxes(title_text='Ground Truth', gridcolor='rgba(128,128,128,0.2)',
                 linecolor='#ccc', showline=True, row=1, col=2)
fig.update_yaxes(title_text='Estimate', gridcolor='rgba(128,128,128,0.2)',
                 linecolor='#ccc', showline=True, row=1, col=2)

# Save
fig.write_image('intelligence_accuracy_plot.png', width=700, height=350, scale=2)
print("Saved: intelligence_accuracy_plot.png")
print(f"Median stated error: {median_upper_error:.1f}%")
print(f"Median estimate error: {median_estimate_error:.1f}%")
