import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Import data from error_bars.py
estimated_quantities = [700, 800, 900, 300, 1000, 50, 800, 441, 18, 1000, 600, 428.0, 287.0, 311.0, 208]
ground_truth_quantities = [610, 280, 847, 0, 1308, 60, 819, 499, 5, 1027.1, 661.2, 347.5, 308.0, 247.5, 287]

stated_error_bars = [
    {"category": "Nuclear Warheads", "min": 150, "max": 160, "possessor": "People's Republic of China"},
    {"category": "Nuclear Warheads", "min": 140, "max": 157, "possessor": "People's Republic of China"},
    {"category": "Nuclear Warheads", "min": 225, "max": 300, "possessor": "People's Republic of China"},
    {"category": "Nuclear Warheads", "min": 1000, "max": 2000, "possessor": "Russian Federation"},
    {"category": "Nuclear Warheads", "min": 60, "max": 80, "possessor": "Pakistan"},
    {"category": "Fissile material (kg)", "min": 25, "max": 35, "possessor": "North Korea"},
    {"category": "Fissile material (kg)", "min": 30, "max": 50, "possessor": "North Korea"},
    {"category": "Fissile material (kg)", "min": 17, "max": 33, "possessor": "North Korea"},
    {"category": "Fissile material (kg)", "min": 335, "max": 400, "possessor": "Pakistan"},
    {"category": "Fissile material (kg)", "min": 330, "max": 580, "possessor": "Israel"},
    {"category": "Fissile material (kg)", "min": 240, "max": 395, "possessor": "India"},
    {"category": "ICBM launchers", "min": 10, "max": 25, "possessor": "Soviet Union"},
    {"category": "ICBM launchers", "min": 10, "max": 25, "possessor": "Soviet Union"},
    {"category": "ICBM launchers", "min": 105, "max": 120, "possessor": "Soviet Union"},
    {"category": "ICBM launchers", "min": 200, "max": 240, "possessor": "Soviet Union"},
    {"category": "Intercontinental missiles", "min": 180, "max": 190, "possessor": "China"},
    {"category": "Intercontinental missiles", "min": 200, "max": 300, "possessor": "Russia"},
    {"category": "Intercontinental missiles", "min": 192, "max": 192, "possessor": "Russia"},
]

# Calculate empirical errors: |estimate - ground_truth|/ground_truth
empirical_errors = []
for est, gt in zip(estimated_quantities, ground_truth_quantities):
    if gt != 0:  # Avoid division by zero
        error = abs(est - gt) / gt
        empirical_errors.append(error)

# Calculate stated errors: (max_val - central_estimate) / central_estimate
stated_errors = []
for entry in stated_error_bars:
    min_val = entry["min"]
    max_val = entry["max"]
    central_estimate = (min_val + max_val) / 2

    if central_estimate != 0:
        error = (max_val - central_estimate) / central_estimate
        stated_errors.append(error)

# Sort the errors and calculate empirical CDF for each data point
# Keep as proportions (0-1)
empirical_errors_sorted = np.sort(empirical_errors)
empirical_cdf = np.arange(1, len(empirical_errors) + 1) / len(empirical_errors)

stated_errors_sorted = np.sort(stated_errors)
stated_cdf = np.arange(1, len(stated_errors) + 1) / len(stated_errors)

# Create square plot
plt.figure(figsize=(5, 5))

# Filter points to only show those within x-axis range [0, 1]
empirical_mask = empirical_errors_sorted <= 1.0
stated_mask = stated_errors_sorted <= 1.0

# Plot empirical errors - one point per data point (filtered)
plt.scatter(empirical_errors_sorted[empirical_mask], empirical_cdf[empirical_mask],
           alpha=0.6, s=40, color='blue', label='Empirical error', zorder=3)

# Plot stated errors - one point per data point (filtered)
plt.scatter(stated_errors_sorted[stated_mask], stated_cdf[stated_mask],
           alpha=0.6, s=40, color='red', label='Stated error', zorder=3)

# Plot lines through all empirical errors (no label for legend)
plt.plot(empirical_errors_sorted, empirical_cdf, 'b-', linewidth=2, zorder=4, alpha=0.8)

# Plot lines through all stated error bars (no label for legend)
plt.plot(stated_errors_sorted, stated_cdf, 'r-', linewidth=2, zorder=4, alpha=0.8)

plt.xlabel('Proportional error')
plt.ylabel('Cumulative probability of error across historical examples')
plt.xlim(0, 1.0)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('diversion_vs_detection.png', dpi=150, bbox_inches='tight')
print("First plot saved as 'diversion_vs_detection.png'")

# Create second plot - only stated error curve with SME labels
plt.figure(figsize=(5, 5))

# Plot only stated errors (no empirical)
plt.scatter(stated_errors_sorted[stated_mask], stated_cdf[stated_mask],
           alpha=0.6, s=40, color='red', label='Stated error margin', zorder=3)

plt.plot(stated_errors_sorted, stated_cdf, 'r-', linewidth=2, zorder=4, alpha=0.8)

# Create logistic regression curve: use all points except the last, then extend to (1.0, 1.0)
# Prepare data for regression
x_reg = list(stated_errors_sorted[:-1])  # All except last point
y_reg = list(stated_cdf[:-1])

# Add endpoint at (1.0, 1.0)
x_reg.append(1.0)
y_reg.append(1.0)

x_reg = np.array(x_reg)
y_reg = np.array(y_reg)

# Fit constrained logistic curve that must pass through origin (0, 0) and plateau at y = 1.0
# Use form: f(x) = 1 - exp(-k * x) which passes through (0, 0) and approaches 1 as x → ∞
from scipy.optimize import curve_fit

def constrained_logistic(x, k):
    # This function passes through (0, 0) and plateaus at y = 1.0
    return 1.0 - np.exp(-k * x)

# Fit the constrained logistic curve
try:
    popt, _ = curve_fit(constrained_logistic, x_reg, y_reg, p0=[5.0], maxfev=5000)
    k_fitted = popt[0]
    x_smooth_reg = np.linspace(0, 1.0, 300)
    y_smooth_reg = constrained_logistic(x_smooth_reg, *popt)
    legend_label = f'$f(x) = 1 - e^{{-{k_fitted:.2f}x}}$'
except:
    # If fitting fails, fall back to simple interpolation
    x_smooth_reg = x_reg
    y_smooth_reg = y_reg
    legend_label = 'Regression curve'

# Plot regression curve
plt.plot(x_smooth_reg, y_smooth_reg, 'g-', linewidth=2, zorder=5, alpha=0.8,
         label=legend_label)

plt.xlabel('Proportion of SME diverted to covert fab', fontsize=12)
plt.ylabel('P(detection | inventory accounting)', fontsize=12)
plt.xlim(0, 1.0)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('sme_diversion_detection.png', dpi=150, bbox_inches='tight')
print("Second plot saved as 'sme_diversion_detection.png'")
