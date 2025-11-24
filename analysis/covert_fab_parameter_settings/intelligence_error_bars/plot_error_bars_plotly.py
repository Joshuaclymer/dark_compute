import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from error_bars import stated_error_bars, estimated_quantities, ground_truth_quantities, categories as estimate_categories, labels

# Website color scheme
COLORS = {
    'purple': '#9B72B0',
    'blue': '#5B8DBE',
    'teal': '#5AA89B',
    'dark_teal': '#2D6B61',
    'red': '#E74C3C',
    'purple_alt': '#8E44AD',
    'light_teal': '#74B3A8'
}

# Calculate central estimates, lower bounds, and upper bounds
central_estimates = []
lower_bounds = []
upper_bounds = []
upper_percent_errors = []
lower_percent_errors = []
categories = []

for entry in stated_error_bars:
    # Skip the Russian Federation nuclear warheads entry (min: 1000, max: 2000)
    if entry.get("possessor") == "Russian Federation" and entry["min"] == 1000 and entry["max"] == 2000:
        continue

    min_val = entry["min"]
    max_val = entry["max"]

    # Central estimate is the midpoint
    central_estimate = (min_val + max_val) / 2
    central_estimates.append(central_estimate)

    # Lower and upper bounds
    lower_bounds.append(min_val)
    upper_bounds.append(max_val)

    # Store category
    categories.append(entry["category"])

    # Calculate percent error for upper bound: (upper_bound - midpoint) / midpoint * 100
    upper_percent_error = ((max_val - central_estimate) / central_estimate) * 100
    upper_percent_errors.append(upper_percent_error)

    # Calculate percent error for lower bound: (midpoint - lower_bound) / midpoint * 100
    lower_percent_error = ((central_estimate - min_val) / central_estimate) * 100
    lower_percent_errors.append(lower_percent_error)

# Calculate median percent errors
median_upper_percent_error = np.median(upper_percent_errors)
median_lower_percent_error = np.median(lower_percent_errors)
print(f"Median upper percent error: {median_upper_percent_error:.2f}%")
print(f"Median lower percent error: {median_lower_percent_error:.2f}%")

# Calculate slopes for regression lines
upper_slope = 1 + (median_upper_percent_error / 100)
lower_slope = 1 - (median_lower_percent_error / 100)
print(f"Upper bound slope: {upper_slope:.4f}")
print(f"Lower bound slope: {lower_slope:.4f}")

# Process estimate vs reality data
estimates = estimated_quantities
ground_truths = ground_truth_quantities

# Calculate percent errors for estimate vs reality
estimate_percent_errors = []
for est, truth in zip(estimates, ground_truths):
    if truth != 0:
        percent_error = abs((est - truth) / truth) * 100
        estimate_percent_errors.append(percent_error)

median_estimate_error = np.median(estimate_percent_errors)
print(f"Median estimate vs reality percent error: {median_estimate_error:.2f}%")

# Calculate slopes for estimate error margin
estimate_upper_slope = 1 + (median_estimate_error / 100)
estimate_lower_slope = 1 - (median_estimate_error / 100)

# Create the figure with two subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Stated ranges', 'Estimate vs. ground truth'),
    horizontal_spacing=0.18
)

# ============= LEFT SUBPLOT: Stated Error Bars =============
# Create a color map for categories using website colors
unique_categories = list(set(categories))
website_colors = [COLORS['purple'], COLORS['blue'], COLORS['teal'],
                 COLORS['dark_teal'], COLORS['purple_alt'], COLORS['light_teal']]
# Extend if needed
while len(website_colors) < len(unique_categories):
    website_colors.extend(website_colors)
category_color_map = {cat: website_colors[i] for i, cat in enumerate(unique_categories)}

# Group data by category for legend
for category in unique_categories:
    # Get indices for this category
    indices = [i for i, c in enumerate(categories) if c == category]

    # Add error bars as lines
    for i in indices:
        fig.add_trace(go.Scatter(
            x=[central_estimates[i], central_estimates[i]],
            y=[lower_bounds[i], upper_bounds[i]],
            mode='lines',
            line=dict(color=category_color_map[category], width=1),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=1)

    # Add upper bound points
    fig.add_trace(go.Scatter(
        x=[central_estimates[i] for i in indices],
        y=[upper_bounds[i] for i in indices],
        mode='markers',
        marker=dict(color=category_color_map[category], size=8),
        opacity=0.7,
        name=category,
        showlegend=True,
        legendgroup=category,
        hovertemplate='Central: %{x}<br>Upper: %{y}<extra></extra>'
    ), row=1, col=1)

    # Add lower bound points
    fig.add_trace(go.Scatter(
        x=[central_estimates[i] for i in indices],
        y=[lower_bounds[i] for i in indices],
        mode='markers',
        marker=dict(color=category_color_map[category], size=8),
        opacity=0.7,
        showlegend=False,
        legendgroup=category,
        hovertemplate='Central: %{x}<br>Lower: %{y}<extra></extra>'
    ), row=1, col=1)

# Set up range for drawing
max_range = max(max(central_estimates), max(upper_bounds))
min_range = min(min(central_estimates), min(lower_bounds))
x_line = np.linspace(min_range, max_range, 100)

# Draw filled region between upper and lower bound trend lines
fig.add_trace(go.Scatter(
    x=np.concatenate([x_line, x_line[::-1]]),
    y=np.concatenate([lower_slope * x_line, (upper_slope * x_line)[::-1]]),
    fill='toself',
    fillcolor='lightgray',
    opacity=0.3,
    line=dict(width=0),
    name=f'Median error margin = {median_upper_percent_error:.1f}%',
    showlegend=True,
    hoverinfo='skip'
), row=1, col=1)

# Draw y = x line (thin grey)
fig.add_trace(go.Scatter(
    x=x_line,
    y=x_line,
    mode='lines',
    line=dict(color='grey', width=1),
    opacity=0.5,
    showlegend=False,
    hoverinfo='skip'
), row=1, col=1)

# ============= RIGHT SUBPLOT: Estimate vs Reality =============
# Create a color map for estimate categories using website colors
unique_estimate_categories = list(set(estimate_categories))
estimate_category_color_map = {cat: website_colors[i] for i, cat in enumerate(unique_estimate_categories)}

# Group data by category for legend
for category in unique_estimate_categories:
    indices = [i for i, c in enumerate(estimate_categories) if c == category]

    fig.add_trace(go.Scatter(
        x=[ground_truths[i] for i in indices],
        y=[estimates[i] for i in indices],
        mode='markers',
        marker=dict(color=estimate_category_color_map[category], size=8),
        opacity=0.7,
        name=category,
        showlegend=True,
        legendgroup=f'est_{category}',
        hovertemplate='Ground truth: %{x}<br>Estimate: %{y}<extra></extra>'
    ), row=1, col=2)

# Set up range for drawing
max_range_est = max(max(estimates), max(ground_truths))
min_range_est = min(min(estimates), min(ground_truths))
x_line_est = np.linspace(min_range_est, max_range_est, 100)

# Draw filled region for median error margin
fig.add_trace(go.Scatter(
    x=np.concatenate([x_line_est, x_line_est[::-1]]),
    y=np.concatenate([estimate_lower_slope * x_line_est, (estimate_upper_slope * x_line_est)[::-1]]),
    fill='toself',
    fillcolor='lightgray',
    opacity=0.3,
    line=dict(width=0),
    name=f'Median error margin = {median_estimate_error:.1f}%',
    showlegend=True,
    hoverinfo='skip'
), row=1, col=2)

# Draw y = x line (thin grey)
fig.add_trace(go.Scatter(
    x=x_line_est,
    y=x_line_est,
    mode='lines',
    line=dict(color='grey', width=1),
    opacity=0.5,
    showlegend=False,
    hoverinfo='skip'
), row=1, col=2)

# Add text labels for specific points
label_offsets = {
    8: (10, 20),  # Missile gap
    1: (10, -25),  # Bomber gap
    3: (10, 40)   # Iraq intelligence failure
}

for label_info in labels:
    idx = label_info["index"]
    label_text = label_info["label"]
    offset = label_offsets.get(idx, (10, 10))

    fig.add_annotation(
        x=ground_truths[idx],
        y=estimates[idx],
        text=label_text,
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor='gray',
        ax=offset[0],
        ay=offset[1],
        font=dict(size=9),
        bgcolor='white',
        bordercolor='gray',
        borderwidth=1,
        borderpad=3,
        opacity=0.8,
        xref='x2',
        yref='y2'
    )

# Update axes labels
fig.update_xaxes(title_text='Central Estimate (Midpoint)', row=1, col=1)
fig.update_yaxes(title_text='Stated estimate range', row=1, col=1)
fig.update_xaxes(title_text='Ground Truth', row=1, col=2)
fig.update_yaxes(title_text='Estimate', row=1, col=2)

# Add grid to both subplots
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=1, col=1)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=1, col=1)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=1, col=2)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=1, col=2)

# Configure separate legends for each subplot
# Left subplot legend
for trace in fig.data:
    if trace.xaxis == 'x':  # Left subplot
        trace.legendgroup = 'left'
        trace.legend = 'legend'

# Right subplot legend
for trace in fig.data:
    if trace.xaxis == 'x2':  # Right subplot
        trace.legendgroup = 'right'
        trace.legend = 'legend2'

# Update layout with two legends
fig.update_layout(
    height=300,
    width=900,
    showlegend=True,
    legend=dict(
        title=dict(text='', font=dict(size=10)),
        font=dict(size=9),
        orientation='v',
        yanchor='top',
        y=1.0,
        xanchor='right',
        x=-0.05,
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='lightgray',
        borderwidth=1
    ),
    legend2=dict(
        title=dict(text='', font=dict(size=10)),
        font=dict(size=9),
        orientation='v',
        yanchor='top',
        y=1.0,
        xanchor='right',
        x=0.45,
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='lightgray',
        borderwidth=1
    ),
    margin=dict(l=140, r=40, t=40, b=50),
    plot_bgcolor='white',
    font=dict(size=10)
)

# Save as HTML for interactive viewing
fig.write_html('../../../static/error_bars_plot_interactive.html')
print("Interactive plot saved as 'error_bars_plot_interactive.html'")

# Save as static image
fig.write_image('../../../static/error_bars_plot.png', width=900, height=300, scale=3)
print("Static plot saved as 'error_bars_plot.png'")
