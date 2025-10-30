"""
MODEL DESCRIPTION (BAYESIAN VERSION):
======================================

This model estimates the detection latency (years from construction start to detection)
for covert nuclear programs as a function of the number of workers with nuclear roles.

REGRESSION MODEL:
The mean detection time is modeled as:
    μ(x) = A / log10(x)^B
where:
    - x = number of workers with nuclear roles
    - A, B = shape parameters controlling the decay rate

PROBABILISTIC MODEL:
Detection times follow a Gamma distribution with:
    - Mean: μ(x) from the regression curve
    - Variance: σ² × μ(x) (proportional to the mean)

For gamma distribution: mean = k*θ, variance = k*θ²
Given μ(x) and variance = σ² × μ(x), we solve for shape k and scale θ:
    k*θ = μ(x)
    k*θ² = σ² × μ(x)
Dividing: θ = σ², k = μ(x) / σ²

BAYESIAN INFERENCE:
We use MCMC (Markov Chain Monte Carlo) to sample from the posterior distribution:
    P(C, A, B, σ² | data) ∝ P(data | C, A, B, σ²) × P(C, A, B, σ²)

Priors (improper uniform):
    - C ~ Uniform(0, ∞)
    - A ~ Uniform(0, ∞)
    - B ~ Uniform(0, ∞)
    - σ² ~ Uniform(0, ∞)

Likelihood:
    y_i ~ Gamma(mean=μ(x_i), variance=σ²)

PREDICTION:
For predictions at new x values, we sample from the posterior predictive distribution:
    1. Sample (C, A, B, σ²) from posterior
    2. Calculate μ(x) = C + A / log10(x)^B
    3. Sample y_new ~ Gamma(mean=μ(x), variance=σ²)
This naturally incorporates parameter uncertainty.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

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
# 2. DEFINE MODEL AND LIKELIHOOD
# ============================================================================

def mean_model(x, A, B):
    """Mean detection time: μ(x) = A / log10(x)^B"""
    return A / (np.log10(x) ** B)

def log_likelihood(params, X, y):
    """
    Log-likelihood for gamma regression
    params = [A, B, sigma_squared]
    """
    A, B, sigma_squared = params

    # Parameter bounds
    if A < 0 or B < 0 or sigma_squared < 0:
        return -np.inf

    try:
        # Calculate means
        mu = mean_model(X, A, B)

        # All means must be positive
        if np.any(mu <= 0):
            return -np.inf

        # Gamma parameters at each data point
        # mean = k*θ = μ, variance = k*θ² = σ²*μ
        # θ = σ², k = μ/σ²
        theta = sigma_squared
        k = mu / sigma_squared

        # All k and theta must be positive
        if np.any(k <= 0) or np.any(theta <= 0):
            return -np.inf

        # Log-likelihood
        log_lik = np.sum(stats.gamma.logpdf(y, a=k, scale=theta))

        return log_lik
    except:
        return -np.inf

def log_prior(params):
    """
    Improper uniform prior (returns 0 if in bounds, -inf otherwise)
    """
    A, B, sigma_squared = params

    # Bounds
    if A < 0 or A > 100:
        return -np.inf
    if B < 0 or B > 10:
        return -np.inf
    if sigma_squared < 0 or sigma_squared > 100:
        return -np.inf

    return 0.0  # Uniform (log of constant)

def log_posterior(params, X, y):
    """Log posterior = log likelihood + log prior"""
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, X, y)

# ============================================================================
# 3. MCMC SAMPLING USING METROPOLIS-HASTINGS
# ============================================================================

print("\n" + "="*60)
print("BAYESIAN INFERENCE USING MCMC")
print("="*60)

# Starting point (use reasonable initial guess)
initial_params = np.array([10.0, 1.5, 6.0])

# Proposal covariance (tuned for good acceptance rate)
proposal_cov = np.diag([0.5, 0.1, 0.5])**2

# MCMC settings
n_samples = 50000
n_burnin = 10000
n_thin = 10

print(f"\nMCMC Settings:")
print(f"  Total samples: {n_samples}")
print(f"  Burn-in: {n_burnin}")
print(f"  Thinning: {n_thin}")
print(f"  Kept samples: {(n_samples - n_burnin) // n_thin}")

# Run MCMC
samples = np.zeros((n_samples, 3))
samples[0] = initial_params
current_log_post = log_posterior(initial_params, X_workers, y)

accepted = 0

print("\nRunning MCMC...")
for i in range(1, n_samples):
    if i % 10000 == 0:
        print(f"  Sample {i}/{n_samples}, acceptance rate: {accepted/i:.3f}")

    # Propose new parameters
    proposal = np.random.multivariate_normal(samples[i-1], proposal_cov)
    proposal_log_post = log_posterior(proposal, X_workers, y)

    # Accept/reject
    log_alpha = proposal_log_post - current_log_post

    if np.log(np.random.rand()) < log_alpha:
        samples[i] = proposal
        current_log_post = proposal_log_post
        accepted += 1
    else:
        samples[i] = samples[i-1]

acceptance_rate = accepted / n_samples
print(f"\nFinal acceptance rate: {acceptance_rate:.3f}")

# Remove burn-in and thin
posterior_samples = samples[n_burnin::n_thin]

print(f"\nPosterior Summary:")
print(f"  A: {np.mean(posterior_samples[:, 0]):.3f} ± {np.std(posterior_samples[:, 0]):.3f}")
print(f"  B: {np.mean(posterior_samples[:, 1]):.3f} ± {np.std(posterior_samples[:, 1]):.3f}")
print(f"  σ²: {np.mean(posterior_samples[:, 2]):.3f} ± {np.std(posterior_samples[:, 2]):.3f}")

print("="*60)

# ============================================================================
# 4. PLOT 1: DETECTION LATENCY VS. NUMBER OF WORKERS
# ============================================================================

# Use posterior mean for main curve
A_mean = np.mean(posterior_samples[:, 0])
B_mean = np.mean(posterior_samples[:, 1])

# Create smooth curve for plotting
X_plot_workers = np.logspace(np.log10(X_workers.min() * 0.8), np.log10(10000), 200)
y_plot = mean_model(X_plot_workers, A_mean, B_mean)

# Calculate 90% credible interval by sampling from posterior predictive
print("\nCalculating posterior predictive intervals...")
y_pred_samples = np.zeros((len(X_plot_workers), len(posterior_samples)))

for i, x_val in enumerate(X_plot_workers):
    for j, (A, B, sigma_sq) in enumerate(posterior_samples):
        mu = mean_model(x_val, A, B)
        if mu > 0 and sigma_sq > 0:
            # Variance proportional to mean: var = σ² × μ
            theta = sigma_sq
            k = mu / sigma_sq
            # Sample from gamma
            y_pred_samples[i, j] = np.random.gamma(k, theta)
        else:
            y_pred_samples[i, j] = 0

# Calculate percentiles
y_lower = np.percentile(y_pred_samples, 5, axis=1)
y_upper = np.percentile(y_pred_samples, 95, axis=1)

plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.size': 14})

# Plot credible interval
plt.fill_between(X_plot_workers, y_lower, y_upper, alpha=0.05, color='blue', label='90% Confidence Interval (shaded)')

# Plot regression line
plt.plot(X_plot_workers, y_plot, 'b-', linewidth=2, label='Posterior Mean')

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
plt.legend(fontsize=12, loc='upper left')

# Parameter box
param_text = f'Bayesian Posterior Mean:\n'
param_text += f'y = {A_mean:.3f} / log10(x)^{B_mean:.3f}\n'
param_text += f'σ² = {np.mean(posterior_samples[:, 2]):.3f}\n'
param_text += f'n = {n}\n'
param_text += f'MCMC samples = {len(posterior_samples)}'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.98, 0.98, param_text, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('detection_latency_vs_workers_bayesian.png', dpi=300, bbox_inches='tight')
print(f"\nPlot 1 saved as 'detection_latency_vs_workers_bayesian.png'")

# ============================================================================
# 5. PLOT 2: P(DETECTED) VS. TIME FOR DIFFERENT WORKER COUNTS
# ============================================================================

plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.size': 14})

# Time range
years = np.linspace(0, 12, 200)

# Worker counts to plot
worker_counts = [100, 1000, 10000]
colors = ['red', 'green', 'blue']
labels = ['100 nuclear workers', '1,000 nuclear workers', '10,000 nuclear workers']
linestyles = ['-', '-', '-']

print("\nCalculating posterior predictive CDFs...")

for workers, color, label, linestyle in zip(worker_counts, colors, labels, linestyles):
    print(f"\n{label}:")

    # Sample predictions from posterior predictive
    p_detected_samples = np.zeros((len(years), len(posterior_samples)))

    for j, (A, B, sigma_sq) in enumerate(posterior_samples):
        mu = mean_model(workers, A, B)

        if mu > 0 and sigma_sq > 0:
            # Variance proportional to mean: var = σ² × μ
            theta = sigma_sq
            k = mu / sigma_sq

            # Calculate CDF at each time point
            for i, t in enumerate(years):
                if t <= 0:
                    p_detected_samples[i, j] = 0.0
                else:
                    p_detected_samples[i, j] = stats.gamma.cdf(t, a=k, scale=theta)

    # Average over posterior samples
    p_detected = np.mean(p_detected_samples, axis=1)

    # Print some values
    idx_3 = np.argmin(np.abs(years - 3))
    idx_5 = np.argmin(np.abs(years - 5))
    print(f"  P(t=3) = {p_detected[idx_3]:.3f}")
    print(f"  P(t=5) = {p_detected[idx_5]:.3f}")
    print(f"  P(t=12) = {p_detected[-1]:.3f}")

    # Plot
    plt.plot(years, p_detected, color=color, linewidth=3.0, linestyle=linestyle, label=label)

# Formatting
plt.xlabel('Years after breaking ground', fontsize=14)
plt.ylabel('Detection probability', fontsize=14)
plt.title('Probability of Detection vs Time for Different Worker Counts', fontsize=14)
plt.xlim(0, 12)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12, loc='lower right')

# Parameter box
param_text_2 = f'Bayesian Model:\n'
param_text_2 += f'μ(x) = {A_mean:.3f} / log10(x)^{B_mean:.3f}\n'
param_text_2 += f'detection_time ~ Gamma(mean=μ(x), variance=σ²)\n'
param_text_2 += f'σ² = {np.mean(posterior_samples[:, 2]):.3f}\n'
param_text_2 += f'n = {n}'

props2 = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, param_text_2, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', horizontalalignment='left', bbox=props2)

plt.tight_layout()
plt.savefig('detection_probability_vs_time_bayesian.png', dpi=300, bbox_inches='tight')
print(f"\nPlot 2 saved as 'detection_probability_vs_time_bayesian.png'")
print("\n" + "="*60)
