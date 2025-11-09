import matplotlib.pyplot as plt
import numpy as np
from error_bars import stated_error_bars, estimated_quantities, ground_truth_quantities, categories as estimate_categories, labels

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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

# ============= LEFT SUBPLOT: Stated Error Bars =============
# Create a color map for categories
unique_categories = list(set(categories))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
category_color_map = {cat: colors[i] for i, cat in enumerate(unique_categories)}

# Draw lines connecting upper and lower bounds for each data point
for i in range(len(central_estimates)):
    color = category_color_map[categories[i]]
    ax1.plot([central_estimates[i], central_estimates[i]],
             [lower_bounds[i], upper_bounds[i]],
             color=color, alpha=0.3, linewidth=1, zorder=1)

# Plot the data points colored by category
plotted_categories = set()
for i in range(len(central_estimates)):
    color = category_color_map[categories[i]]
    label = categories[i] if categories[i] not in plotted_categories else None
    if label:
        plotted_categories.add(categories[i])
    ax1.scatter(central_estimates[i], upper_bounds[i], alpha=0.7, s=50, color=color, label=label, zorder=3)
    ax1.scatter(central_estimates[i], lower_bounds[i], alpha=0.7, s=50, color=color, zorder=3)

# Set up range for drawing
max_range = max(max(central_estimates), max(upper_bounds))
min_range = min(min(central_estimates), min(lower_bounds))
x_line = np.linspace(min_range, max_range, 100)

# Draw y = x line (thin grey, not in legend)
ax1.plot(x_line, x_line, color='grey', linewidth=1, alpha=0.5, zorder=0)

# Draw light gray filled region between upper and lower bound trend lines
ax1.fill_between(x_line, lower_slope * x_line, upper_slope * x_line,
                 color='lightgray', alpha=0.3, label=f'Median error margin = {median_upper_percent_error:.1f}%', zorder=1)

# Labels and title
ax1.set_xlabel('Central Estimate (Midpoint)', fontsize=12)
ax1.set_ylabel('Stated estimate range', fontsize=12)
ax1.set_title('Stated ranges', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3, which='both')

# ============= RIGHT SUBPLOT: Estimate vs Reality =============
# Create a color map for estimate categories
unique_estimate_categories = list(set(estimate_categories))
estimate_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_estimate_categories)))
estimate_category_color_map = {cat: estimate_colors[i] for i, cat in enumerate(unique_estimate_categories)}

# Plot the data points colored by category
plotted_estimate_categories = set()
for i in range(len(estimates)):
    color = estimate_category_color_map[estimate_categories[i]]
    label = estimate_categories[i] if estimate_categories[i] not in plotted_estimate_categories else None
    if label:
        plotted_estimate_categories.add(estimate_categories[i])
    ax2.scatter(ground_truths[i], estimates[i], alpha=0.7, s=50, color=color, label=label, zorder=3)

# Set up range for drawing
max_range_est = max(max(estimates), max(ground_truths))
min_range_est = min(min(estimates), min(ground_truths))
x_line_est = np.linspace(min_range_est, max_range_est, 100)

# Draw y = x line (thin grey, not in legend)
ax2.plot(x_line_est, x_line_est, color='grey', linewidth=1, alpha=0.5, zorder=0)

# Draw light gray filled region for median error margin
ax2.fill_between(x_line_est, estimate_lower_slope * x_line_est, estimate_upper_slope * x_line_est,
                 color='lightgray', alpha=0.3, label=f'Median error margin = {median_estimate_error:.1f}%', zorder=1)

# Add text labels for specific points with custom positioning
label_offsets = {
    8: (10, 20),  # Missile gap - higher
    1: (10, -25),  # Bomber gap - lower
    3: (10, 40)   # Iraq intelligence failure - even higher
}

for label_info in labels:
    idx = label_info["index"]
    label_text = label_info["label"]
    offset = label_offsets.get(idx, (10, 10))
    ax2.annotate(label_text,
                xy=(ground_truths[idx], estimates[idx]),
                xytext=offset, textcoords='offset points',
                fontsize=9, alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='gray', lw=0.5))

# Labels and title
ax2.set_xlabel('Ground Truth', fontsize=12)
ax2.set_ylabel('Estimate', fontsize=12)
ax2.set_title('Estimate vs. ground truth', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()

# Save the plot
plt.savefig('stated_error_bars_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'stated_error_bars_plot.png'")
