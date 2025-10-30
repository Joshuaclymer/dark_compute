import matplotlib.pyplot as plt
import numpy as np

wafers_per_month = [4000, 55000, 31500, 800000, 83000, 20000, 33000, 62500, 100000, 140000, 450000]

employees = [80, 1500, 3950, 31000, 11300, 2200, 2750, 3000, 3000, 8500, 10000]

# Divide employees by 3
employees_adjusted = [e / 3 for e in employees]

# Create plot
plt.figure(figsize=(10, 6))

# Add labels and title
plt.xlabel('Number of Workers', fontsize=12)
plt.ylabel('Wafers per Month', fontsize=12)
plt.title('Operating Labor vs Wafers Production (Log-Log Scale)', fontsize=14, fontweight='bold')

# Set log scale for both axes
plt.xscale('log')
plt.yscale('log')

# Set axis limits - flipped
plt.xlim(50, 20000)
plt.ylim(1000, 1000000)

# Set specific tick marks for better readability
# X-axis (workers) - specific values from 10 to 20000
x_ticks = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
x_labels = [str(x) if x < 1000 else f'{int(x/1000)}K' for x in x_ticks]
plt.xticks(x_ticks, x_labels)

# Y-axis (wafers per month) - from 1,000 to 1,000,000
y_ticks = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
plt.yticks(y_ticks, [f'{int(y/1000)}K' if y < 1000000 else f'{int(y/1000000)}M' for y in y_ticks])

# Add grid for better readability
plt.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.15, which='minor', linestyle=':', linewidth=0.3)

# Add a trend line for operating employees (blue, in log space) - flipped axes
log_employees = np.log10(employees_adjusted)
log_wafers = np.log10(wafers_per_month)
z_ops = np.polyfit(log_wafers, log_employees, 1)
p_ops = np.poly1d(z_ops)
# For flipped axes: x is now workers, y is now wafers
# Original: log(workers) = z_ops[1] + z_ops[0]*log(wafers)
# Inverted: log(wafers) = (log(workers) - z_ops[1]) / z_ops[0]
x_range_ops = np.logspace(np.log10(min(employees_adjusted)), np.log10(max(employees_adjusted)), 100)
y_range_ops = 10**((np.log10(x_range_ops) - z_ops[1]) / z_ops[0])
plt.plot(x_range_ops, y_range_ops, "--", alpha=0.7, color='blue', linewidth=2,
         label=f'Point estimates ({z_ops[1]:.2f} + {z_ops[0]:.2f}*log10(x))')

plt.legend(loc='best', fontsize=10)
plt.tight_layout()
plt.savefig('operating_labor_vs_production.png', dpi=300, bbox_inches='tight')

print(f"Plot saved as 'operating_labor_vs_production.png'")
print(f"\nData points: {len(wafers_per_month)}")
print(f"Wafers/Employee ratio range: {min([w/e for w, e in zip(wafers_per_month, employees_adjusted)]):.1f} - {max([w/e for w, e in zip(wafers_per_month, employees_adjusted)]):.1f}")
