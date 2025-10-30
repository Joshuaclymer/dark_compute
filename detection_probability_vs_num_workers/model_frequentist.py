"""
MODEL DESCRIPTION:
==================

This model estimates the detection latency (years from construction start to detection)
for covert nuclear programs as a function of the number of workers with nuclear roles.

REGRESSION MODEL:
The mean detection time is modeled as:
    μ(x) = C + A / log10(x)^B
where:
    - x = number of workers with nuclear roles
    - C = floor/plateau parameter (minimum detection time as x → ∞)
    - A, B = shape parameters controlling the decay rate

PROBABILISTIC MODEL:
Detection times follow a Gamma distribution with:
    - Mean: μ(x) from the regression curve
    - Variance: σ² (constant across all x values)

For gamma distribution: mean = k*θ, variance = k*θ²
Given μ(x) and σ², we can solve for shape k and scale θ:
    θ = σ²/μ(x)
    k = μ(x)/θ = μ(x)²/σ²

This means gamma distributions at different x values have different shapes,
but all have the same variance.

PARAMETER ESTIMATION:
- Stage 1: Fit mean function μ(x) = C + A / log10(x)^B using nonlinear least squares
- Stage 2: Estimate σ² from the residual variance

P(DETECTION) CURVES:
For a given number of workers x:
1. Calculate μ(x) using the fitted regression parameters
2. Create a Gamma(k, θ) distribution and shift it so its mean = μ(x)
3. Truncate this distribution at detection_time = 0 (remove negative mass, no renormalization)
4. Use the CDF of this truncated distribution as P(detected by time t)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

# Read the data
df = pd.read_csv('nuclear_case_studies.csv')

# Parse detection latency
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

# Parse worker numbers
def extract_workers(text):
    """Extract numeric worker count"""
    if pd.isna(text):
        return np.nan
    try:
        import re
        match = re.search(r'[\d,]+', str(text))
        if match:
            return float(match.group().replace(',', ''))
    except:
        pass
    return np.nan

df['detection_years'] = df['Function identified'].apply(extract_years)
df['num_workers'] = df['Workers (nuclear roles)'].apply(extract_workers)

# Clean data
df_clean = df[['Site', 'num_workers', 'detection_years']].dropna()
print(f"Analyzing {len(df_clean)} facilities with complete data:")
print(df_clean)

X_workers = df_clean['num_workers'].values
y = df_clean['detection_years'].values
n = len(y)

# ============================================================================
# 2. FIT MODEL USING TWO-STAGE APPROACH
# ============================================================================

def mean_model(x, C, A, B):
    """Mean detection time: μ(x) = C + A / log10(x)^B"""
    return C + A / (np.log10(x) ** B)

def gradient_mean(x, C, A, B):
    """
    Gradient of mean_model with respect to parameters [C, A, B]
    Used for calculating prediction variance via delta method
    """
    log_x = np.log10(x)
    log_x_B = log_x ** B

    # ∂μ/∂C = 1
    dmu_dC = 1.0

    # ∂μ/∂A = 1 / log10(x)^B
    dmu_dA = 1.0 / log_x_B

    # ∂μ/∂B = -A * log10(x)^B * ln(log10(x)) / log10(x)^(2B)
    #        = -A * ln(log10(x)) / log10(x)^B
    dmu_dB = -A * np.log(log_x) / log_x_B

    return np.array([dmu_dC, dmu_dA, dmu_dB])

def prediction_variance(x, C, A, B, param_cov, sigma_squared):
    """
    Calculate prediction variance at point x using delta method
    Var(y_new) = sigma_squared + grad^T * Cov * grad
    """
    grad = gradient_mean(x, C, A, B)
    var_mean = grad @ param_cov @ grad  # Variance of the mean estimate
    return sigma_squared + var_mean

print("\n" + "="*60)
print("FITTING MODEL WITH TWO-STAGE APPROACH")
print("="*60)

# STAGE 1: Fit mean function using nonlinear least squares
print("\nSTAGE 1: Fitting mean function μ(x) = C + A / log10(x)^B")
print("Using nonlinear least squares (curve_fit)...")

from scipy.optimize import curve_fit

# Initial guess for mean parameters
initial_mean = [2.0, 10.0, 2.0]

# Bounds for mean parameters
bounds_mean = ([0, 0.1, 0.5], [10, 50, 10])

try:
    popt, pcov = curve_fit(mean_model, X_workers, y,
                           p0=initial_mean,
                           bounds=bounds_mean,
                           maxfev=10000)
    C_param, A_param, B_param = popt
    param_cov = pcov  # Save covariance matrix for later use

    print(f"  C (floor)     = {C_param:.4f}")
    print(f"  A (scale)     = {A_param:.4f}")
    print(f"  B (power)     = {B_param:.4f}")

    # Calculate R-squared
    y_pred = mean_model(X_workers, C_param, A_param, B_param)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"  R²            = {r_squared:.4f}")

except Exception as e:
    print(f"  Curve fit failed: {e}")
    print("  Using fallback parameters...")
    C_param, A_param, B_param = 1.878, 12.910, 2.206
    param_cov = np.diag([0.1, 1.0, 0.1])  # Rough estimate
    y_pred = mean_model(X_workers, C_param, A_param, B_param)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"  R² = {r_squared:.4f}")

# STAGE 2: Estimate variance from residuals
print("\nSTAGE 2: Estimating variance from residuals")

# Calculate residuals
residuals = y - y_pred

print(f"  Residual mean = {np.mean(residuals):.4f}")
print(f"  Residual std  = {np.std(residuals, ddof=1):.4f}")

# Estimate variance
residual_variance = np.var(residuals, ddof=1)
sigma_squared = residual_variance

print(f"  σ² (variance) = {sigma_squared:.4f}")
print(f"  σ (std dev)   = {np.sqrt(sigma_squared):.4f}")

print("="*60)

# ============================================================================
# 3. PLOT 1: DETECTION LATENCY VS. NUMBER OF WORKERS
# ============================================================================

# Create smooth curve for plotting
X_plot_workers = np.logspace(np.log10(X_workers.min() * 0.8), np.log10(10000), 200)
y_plot = mean_model(X_plot_workers, C_param, A_param, B_param)

# Calculate proper 90% prediction interval using gamma distribution
# For each x value, gamma has mean μ(x) and prediction variance σ²_pred

y_lower = np.zeros_like(y_plot)
y_upper = np.zeros_like(y_plot)

for i, (x_val, mu) in enumerate(zip(X_plot_workers, y_plot)):
    # Calculate prediction variance (residual + parameter uncertainty)
    var_pred = prediction_variance(x_val, C_param, A_param, B_param, param_cov, sigma_squared)

    # Calculate gamma parameters for this mean and prediction variance
    # mean = k*θ = μ, variance = k*θ² = var_pred
    # Therefore: θ = var_pred/μ, k = μ²/var_pred

    if mu > 0 and var_pred > 0:
        theta_i = var_pred / mu
        k_i = mu / theta_i

        # 5th and 95th percentiles
        y_lower[i] = stats.gamma.ppf(0.05, a=k_i, scale=theta_i)
        y_upper[i] = stats.gamma.ppf(0.95, a=k_i, scale=theta_i)
    else:
        y_lower[i] = 0
        y_upper[i] = 0

plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.size': 14})

# Plot confidence interval
plt.fill_between(X_plot_workers, y_lower, y_upper, alpha=0.05, color='blue', label='90% Prediction Interval')

# Plot regression line
plt.plot(X_plot_workers, y_plot, 'b-', linewidth=2, label='Regression Line')

# Plot data points with labels
for idx, row in df_clean.iterrows():
    plt.plot(row['num_workers'], row['detection_years'], 'kx', markersize=10, markeredgewidth=2)

# Add labels with smart positioning
points_data = [(row['num_workers'], row['detection_years'], row['Site'])
               for _, row in df_clean.iterrows()]
points_data.sort(key=lambda p: (np.log10(p[0]), p[1]))

label_positions = []
for x_pos, y_pos, site_name in points_data:
    offset_y = y_pos

    max_iterations = 100
    for iteration in range(max_iterations):
        conflict = False
        for prev_x, prev_y in label_positions:
            x_log_dist = abs(np.log10(x_pos) - np.log10(prev_x))
            y_dist = abs(offset_y - prev_y)

            if x_log_dist < 0.6 and y_dist < 1.8:
                conflict = True
                offset_y += 0.9
                break

        if not conflict:
            break

    label_positions.append((x_pos, offset_y))

    if abs(offset_y - y_pos) > 0.4:
        plt.plot([x_pos, x_pos * 1.22], [y_pos, offset_y], 'k-', linewidth=0.5, alpha=0.4)

    plt.text(x_pos * 1.25, offset_y, site_name, fontsize=9, verticalalignment='center', ha='left')

# Formatting
plt.xscale('log')
plt.ylim(bottom=0)
plt.xticks([100, 1000, 10000], ['100', '1,000', '10,000'])
plt.xlabel('Number of Nuclear Role Workers (log scale)', fontsize=14)
plt.ylabel('Detection Latency (years)', fontsize=14)
plt.title('Detection Latency vs. Number of Workers in Covert Nuclear Programs', fontsize=14)
plt.grid(True, alpha=0.3)

# Parameter box
param_text = f'Regression Parameters:\n'
param_text += f'y = {C_param:.3f} + {A_param:.3f} / log10(x)^{B_param:.3f}\n'
param_text += f'R^2 = {r_squared:.3f}\n'
param_text += f'n = {n}\n'
param_text += f'σ² = {sigma_squared:.3f}'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.98, 0.98, param_text, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('detection_latency_vs_workers.png', dpi=300, bbox_inches='tight')
print(f"\nPlot 1 saved as 'detection_latency_vs_workers.png'")

# ============================================================================
# 4. PLOT 2: P(DETECTED) VS. TIME FOR DIFFERENT WORKER COUNTS
# ============================================================================

plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.size': 14})

# Time range
years = np.linspace(0, 12, 200)

# Worker counts to plot
worker_counts = [100, 1000, 10000]
colors = ['red', 'green', 'blue']
labels = ['100 workers', '1,000 workers', '10,000 workers']
linestyles = ['-', '--', '-']

for workers, color, label, linestyle in zip(worker_counts, colors, labels, linestyles):
    # Calculate mean for this worker count
    mu = mean_model(workers, C_param, A_param, B_param)

    # Calculate prediction variance (includes parameter uncertainty)
    var_pred = prediction_variance(workers, C_param, A_param, B_param, param_cov, sigma_squared)

    # Calculate gamma parameters: mean = μ, variance = var_pred
    # θ = var_pred/μ, k = μ²/var_pred
    theta_i = var_pred / mu
    k_i = mu / theta_i

    print(f"\nWorkers: {workers}, μ = {mu:.3f}, σ²_pred = {var_pred:.3f}, k = {k_i:.3f}, θ = {theta_i:.3f}")

    # Calculate P(detected by t) using the gamma CDF
    p_detected = np.zeros_like(years)

    for i, t in enumerate(years):
        if t <= 0:
            p_detected[i] = 0.0
        else:
            # Simple gamma CDF - no shifting or truncation needed
            p_detected[i] = stats.gamma.cdf(t, a=k_i, scale=theta_i)

    print(f"  P(t=3) = {p_detected[np.argmin(np.abs(years - 3))]:.3f}")
    print(f"  P(t=5) = {p_detected[np.argmin(np.abs(years - 5))]:.3f}")
    print(f"  P(t=12) = {p_detected[-1]:.3f}")

    # Plot
    plt.plot(years, p_detected, color=color, linewidth=3.0, linestyle=linestyle, label=label)

# Formatting
plt.xlabel('Years into agreement', fontsize=14)
plt.ylabel('Annual detection probability', fontsize=14)
plt.title('Probability of Detection vs Time for Different Worker Counts', fontsize=14)
plt.xlim(0, 12)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12, loc='lower right')

# Parameter box
param_text_2 = f'Model:\n'
param_text_2 += f'μ(x) = {C_param:.3f} + {A_param:.3f} / log10(x)^{B_param:.3f}\n'
param_text_2 += f'detection_time ~ Gamma(mean=μ(x), variance=σ²)\n'
param_text_2 += f'σ² = {sigma_squared:.3f}\n'
param_text_2 += f'n = {n}'

props2 = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, param_text_2, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', horizontalalignment='left', bbox=props2)

plt.tight_layout()
plt.savefig('detection_probability_vs_time.png', dpi=300, bbox_inches='tight')
print(f"\nPlot 2 saved as 'detection_probability_vs_time.png'")
print("\n" + "="*60)
