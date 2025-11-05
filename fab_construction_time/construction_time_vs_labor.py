import matplotlib.pyplot as plt
import numpy as np

# Constants from the model
WORKERS_PER_BILLION_USD = 100  # Concurrent workers needed per $1B USD

# Regression coefficient from cost_of_fab_plot.py
# cost = alpha * capacity, where capacity is in wafers/month and cost is in billions USD
COST_PER_CAPACITY = 0.000141  # This should match the alpha from cost_of_fab_plot.py

# Regression coefficients from construction_time_plot.py (with concealment)
# construction_time = slope * log10(capacity) + intercept
# These are from the 1.5x adjusted relationship (with concealment)
SLOPE_CONCEALMENT = 0.773  # From z_concealment[0]
INTERCEPT_CONCEALMENT = -1.456  # From z_concealment[1]

def estimate_cost(capacity_wafers_per_month):
    """Estimate fab cost in billions USD based on production capacity"""
    return COST_PER_CAPACITY * capacity_wafers_per_month

def required_labor(capacity_wafers_per_month):
    """Calculate required concurrent workers for standard construction timeline"""
    cost_billion = estimate_cost(capacity_wafers_per_month)
    return WORKERS_PER_BILLION_USD * cost_billion

def baseline_construction_time(capacity_wafers_per_month):
    """
    Calculate baseline construction time in years based on capacity,
    using the adjusted relationship with concealment from construction_time_plot.py
    """
    log_capacity = np.log10(capacity_wafers_per_month)
    return SLOPE_CONCEALMENT * log_capacity + INTERCEPT_CONCEALMENT

def construction_time(capacity_wafers_per_month, actual_labor):
    """
    Calculate construction time in years based on labor supply.

    If labor < required: construction time increases proportionally
    If labor >= required: construction time stays at baseline (no speedup)
    """
    baseline_time = baseline_construction_time(capacity_wafers_per_month)
    req_labor = required_labor(capacity_wafers_per_month)

    if actual_labor < req_labor:
        # Understaffed: time increases proportionally
        return baseline_time * (req_labor / actual_labor)
    else:
        # Adequately staffed or overstaffed: time stays at baseline
        return baseline_time

# Define three different production capacities to plot
capacities = [5000, 25000, 120000]  # wafers per month
capacity_labels = ['5k wafers/month', '25k wafers/month', '120k wafers/month']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Create labor range (0 to 3x the maximum required labor)
max_capacity = max(capacities)
max_required = required_labor(max_capacity)
labor_range = np.linspace(1, 3 * max_required, 500)

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

# Plot construction time vs labor for each capacity
for capacity, label, color in zip(capacities, capacity_labels, colors):
    req_labor = required_labor(capacity)
    times = [construction_time(capacity, labor) for labor in labor_range]

    ax.plot(labor_range, times, linewidth=2, label=label, color=color)

    # Mark the required labor point
    baseline_time = construction_time(capacity, req_labor)
    ax.plot(req_labor, baseline_time, 'o', markersize=8, color=color,
            markeredgecolor='black', markeredgewidth=1)

# Note: No single horizontal baseline line since each capacity has different baseline time

ax.set_xlabel('Construction labor', fontsize=12)
ax.set_ylabel('Construction Time (Years)', fontsize=12)

# Set y-axis limit to show reasonable range
ax.set_ylim(bottom=0, top=15)
ax.set_xlim(left=0)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('construction_time_vs_labor.png', dpi=300, bbox_inches='tight')

print(f"Plot saved as 'construction_time_vs_labor.png'")
print(f"\nModel parameters:")
print(f"  Construction time relationship (with concealment):")
print(f"    y = {SLOPE_CONCEALMENT:.3f}*log10(capacity) + {INTERCEPT_CONCEALMENT:.3f}")
print(f"  Workers per $1B USD: {WORKERS_PER_BILLION_USD}")
print(f"  Cost coefficient: ${COST_PER_CAPACITY:.6f}B per wafer/month capacity")
print(f"\nRequired labor and baseline time for each capacity:")
for capacity, label in zip(capacities, capacity_labels):
    req = required_labor(capacity)
    cost = estimate_cost(capacity)
    baseline_time = baseline_construction_time(capacity)
    print(f"  {label}:")
    print(f"    Required workers: {req:.0f} (estimated cost: ${cost:.2f}B)")
    print(f"    Baseline construction time: {baseline_time:.2f} years")
