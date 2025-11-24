"""
Generate detection latency vs workers plot in original matplotlib style
Uses cached Monte Carlo simulation data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load the saved Bayesian model data
with open('model_bayesian_data.pkl', 'rb') as f:
    saved_data = pickle.load(f)

posterior_samples = saved_data['posterior_samples']

# Calculate posterior means
A_mean = np.mean(posterior_samples[:, 0])
B_mean = np.mean(posterior_samples[:, 1])

# Load the cached plot data
with open('model_bayesian_plot_data.pkl', 'rb') as f:
    plot_data = pickle.load(f)

X_plot_workers = plot_data['X_plot_workers']
y_plot = plot_data['y_plot']
y_lower = plot_data['y_lower']
y_upper = plot_data['y_upper']

# Load and prepare the actual data
df = pd.read_csv('nuclear_case_studies.csv')

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

def extract_workers(text):
    """Extract numeric worker count"""
    if pd.isna(text):
        return np.nan
    try:
        import re
        text_clean = str(text).replace(',', '').replace('~', '')
        match = re.search(r'(\d+)', text_clean)
        if match:
            return float(match.group(1))
    except:
        pass
    return np.nan

df['detection_years'] = df['Function identified'].apply(extract_years)
df['num_workers'] = df['Workers (nuclear roles)'].apply(extract_workers)
df_clean = df.dropna(subset=['detection_years', 'num_workers'])

# Create figure
plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 11})

# Plot credible interval
plt.fill_between(X_plot_workers, y_lower, y_upper, alpha=0.2, color='#5B8DBE', label='90% Confidence Interval')

# Plot regression line
plt.plot(X_plot_workers, y_plot, '-', color='#5B8DBE', linewidth=2, label='Posterior Mean')

# Plot data points
for idx, row in df_clean.iterrows():
    plt.plot(row['num_workers'], row['detection_years'], 'o', color='#5B8DBE', markersize=8,
             markeredgewidth=1, markeredgecolor='white')

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
plt.xlabel('Nuclear-role workers', fontsize=11)
plt.ylabel('Detection latency (years)', fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc='upper left')

plt.tight_layout()
plt.savefig('detection_latency_vs_workers_bayesian.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'detection_latency_vs_workers_bayesian.png'")
plt.close()
