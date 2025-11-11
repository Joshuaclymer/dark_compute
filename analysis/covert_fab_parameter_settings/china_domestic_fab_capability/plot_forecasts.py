import matplotlib.pyplot as plt
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

# Create the plot
fig, ax = plt.subplots(figsize=(8, 5))

# Colors for each node
colors = {
    '130nm': '#6A994E',  # Green
    '28nm': '#2E86AB',   # Blue
    '14nm': '#A23B72',   # Purple
    '7nm': '#F18F01'     # Orange
}

# Plot each node's forecast
for node, data in forecasts.items():
    # Extract years and probabilities
    years = np.array([point['year'] for point in data])
    probabilities = np.array([point['probability'] for point in data])

    # Create smooth curve using quadratic interpolation (perfect for 3 points)
    # Generate more points for a smooth curve
    years_smooth = np.linspace(years.min(), years.max(), 100)

    # Use quadratic interpolation for smooth curves
    interpolator = interp1d(years, probabilities, kind='quadratic')
    probabilities_smooth = interpolator(years_smooth)

    # Plot the smooth curve
    ax.plot(years_smooth, probabilities_smooth,
            label=node,
            color=colors[node],
            linewidth=2.0)

    # Plot the original data points
    ax.scatter(years, probabilities,
              color=colors[node],
              s=80,
              zorder=5,
              edgecolors='white',
              linewidth=1.5)

# Styling
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Probability of >90% localization for node', fontsize=12)
ax.set_title('China Semiconductor Self-Sufficiency Probability Forecasts',
             fontsize=14, pad=20)

# Set y-axis to percentage format
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Legend
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)

# Set axis limits
ax.set_xlim(2024.5, 2031.5)
ax.set_ylim(0, 1)

# Tight layout
plt.tight_layout()

# Save the plot
plt.savefig('self_sufficiency_forecasts.png', dpi=300, bbox_inches='tight')
print("Forecast plot saved to self_sufficiency_forecasts.png")

# Also create a version with shaded confidence regions
fig2, ax2 = plt.subplots(figsize=(8, 5))

for node, data in forecasts.items():
    years = np.array([point['year'] for point in data])
    probabilities = np.array([point['probability'] for point in data])

    years_smooth = np.linspace(years.min(), years.max(), 100)
    interpolator = interp1d(years, probabilities, kind='quadratic')
    probabilities_smooth = interpolator(years_smooth)

    # Plot the smooth curve
    ax2.plot(years_smooth, probabilities_smooth,
            label=node,
            color=colors[node],
            linewidth=2.0)

    # Add shaded area under the curve
    ax2.fill_between(years_smooth, 0, probabilities_smooth,
                     alpha=0.2,
                     color=colors[node])

    # Plot the original data points
    ax2.scatter(years, probabilities,
              color=colors[node],
              s=80,
              zorder=5,
              edgecolors='white',
              linewidth=1.5)

# Styling
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Probability of Self-Sufficiency', fontsize=12)
ax2.set_title('China Semiconductor Self-Sufficiency Probability Forecasts',
             fontsize=14, pad=20)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax2.legend(loc='upper left', fontsize=11, framealpha=0.95)
ax2.set_xlim(2024.5, 2031.5)
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('self_sufficiency_forecasts_shaded.png', dpi=300, bbox_inches='tight')
print("Shaded forecast plot saved to self_sufficiency_forecasts_shaded.png")

# Create inverted version (1 - probability) for detection probability
fig3, ax3 = plt.subplots(figsize=(5, 5))

for node, data in forecasts.items():
    years = np.array([point['year'] for point in data])
    probabilities = np.array([point['probability'] for point in data])

    # Invert probabilities (1 - p)
    probabilities_inverted = 1 - probabilities

    years_smooth = np.linspace(years.min(), years.max(), 100)
    interpolator = interp1d(years, probabilities_inverted, kind='quadratic')
    probabilities_smooth = interpolator(years_smooth)

    # Plot the smooth curve
    ax3.plot(years_smooth, probabilities_smooth,
            label=node,
            color=colors[node],
            linewidth=2.0)

    # Plot the original data points
    ax3.scatter(years, probabilities_inverted,
              color=colors[node],
              s=80,
              zorder=5,
              edgecolors='white',
              linewidth=1.5)

# Styling
ax3.set_xlabel('Year', fontsize=12)
ax3.set_ylabel('P(detection | procurement accounting)', fontsize=12)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax3.legend(loc='center left', fontsize=11, framealpha=0.95)
ax3.set_xlim(2024.5, 2031.5)
ax3.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('detection_probability_forecasts.png', dpi=300, bbox_inches='tight')
print("Detection probability forecast plot saved to detection_probability_forecasts.png")
