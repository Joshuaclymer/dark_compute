import matplotlib.pyplot as plt
import numpy as np

# Create data for the plot
# x-axis: number of photolithography scanners (from 0 to 500)
scanners = np.array([0, 500])

# y-axis: wafer starts per month (WSPM)
# Relationship: WSPM = 1000 * scanners
wspm = 1000 * scanners

# Create the plot (smaller, non-square dimensions)
plt.figure(figsize=(8, 5))
plt.plot(scanners, wspm, '-', linewidth=2, label='y = 1000 * x')

# Set axis limits
plt.xlim(0, 500)
plt.ylim(0, 500000)

# Add grid
plt.grid(True, alpha=0.3)

# Labels and title
plt.xlabel('Number of Photolithography Scanners', fontsize=12)
plt.ylabel('Wafer Starts Per Month (WSPM)', fontsize=12)
plt.title('Production vs SME: Relationship Between Scanners and WSPM', fontsize=14, fontweight='bold')

# Add legend
plt.legend(fontsize=11, loc='upper left')

# Save the plot
plt.tight_layout()
plt.savefig('production_vs_sme.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'production_vs_sme.png'")

