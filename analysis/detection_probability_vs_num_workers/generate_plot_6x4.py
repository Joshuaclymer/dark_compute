"""
Generate a 6x4 version of detection_probability_vs_time_bayesian.png
using the same plotting style as the web app.

This script uses cached data from model_bayesian_plot_data.pkl
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load cached plot data
print("Loading cached plot data...")
with open('model_bayesian_plot_data.pkl', 'rb') as f:
    plot_data = pickle.load(f)

years = plot_data['years']
worker_counts = plot_data['worker_counts']
p_detected_dict = plot_data['p_detected_dict']

print(f"Data loaded. Generating plot...")

# Create figure with 6x3.5 size
plt.figure(figsize=(6, 3.5))
plt.rcParams.update({'font.size': 10})

# Colors and labels - matching the web app style
colors = ['#9B7BB3', '#5B8DBE', '#5AA89B']  # Purple, Blue, Blue-green (from web app)
labels = ['100 high-context workers involved', '1,000 high-context workers involved', '10,000 high-context workers involved']
linestyles = ['-', '-', '-']

# Plot each worker count
for workers, color, label, linestyle in zip(worker_counts, colors, labels, linestyles):
    p_detected = p_detected_dict[workers]
    plt.plot(years, p_detected, color=color, linewidth=2.0, linestyle=linestyle, label=label)

# Formatting - matching web app style
plt.xlabel('Years after breaking ground', fontsize=10)
plt.ylabel('P(strong evidence of covert project)', fontsize=10)
plt.xlim(0, 12)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=9, loc='lower right', frameon=False)

plt.tight_layout()
plt.savefig('detection_probability_vs_time_bayesian_6x4.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'detection_probability_vs_time_bayesian_6x4.png'")
