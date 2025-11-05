import matplotlib.pyplot as plt
import numpy as np

fabs = [
    {
        "company": "SMIC-SMSC",
        "facility": "SN1",
        "location": "China, Shanghai",
        "cost_billion_usd": 10.0,
        "start_year": 2020,
        "wafer_size_mm": 300,
        "node_nm": 14,
        "capacity_wafers": 70000,
        "technology": "Foundry"
    },
    {
        "company": "SMIC",
        "facility": "B2A",
        "location": "China, Beijing",
        "cost_billion_usd": 3.59,
        "start_year": 2014,
        "wafer_size_mm": 300,
        "node_nm": 28,
        "capacity_wafers": 41000,
        "technology": "Foundry"
    },
    {
        "company": "GlobalFoundries",
        "facility": "Fab 8",
        "location": "USA, New York, Malta",
        "cost_billion_usd": 6.7,
        "start_year": 2012,
        "wafer_size_mm": 300,
        "node_nm": 12,
        "capacity_wafers": 60000,
        "technology": "Foundry"
    },
    {
        "company": "TSMC",
        "facility": "Fab 18 (P1-3)",
        "location": "Taiwan, Tainan",
        "cost_billion_usd": 17.0,
        "start_year": 2018,
        "wafer_size_mm": 300,
        "node_nm": 5,
        "capacity_wafers": 120000,
        "technology": "Foundry"
    },
    {
        "company": "TSMC",
        "facility": "Fab 15",
        "location": "Taiwan, Taichung",
        "cost_billion_usd": 9.3,
        "start_year": 2011,
        "wafer_size_mm": 300,
        "node_nm": 20,
        "capacity_wafers": 100000,
        "technology": "Foundry"
    },
    {
        "company": "TSMC",
        "facility": "Fab 14 (P4)",
        "location": "Taiwan, Tainan",
        "cost_billion_usd": 3.75,
        "start_year": 2011,
        "wafer_size_mm": 300,
        "node_nm": 16,
        "capacity_wafers": 45500,
        "technology": "Foundry"
    },
    {
        "company": "TSMC",
        "facility": "Fab 12 (P4)",
        "location": "Taiwan, Hsinchu",
        "cost_billion_usd": 6.0,
        "start_year": 2009,
        "wafer_size_mm": 300,
        "node_nm": 20,
        "capacity_wafers": 40000,
        "technology": "Foundry"
    },
    {
        "company": "TSMC",
        "facility": "Fab 14 (P3)",
        "location": "Taiwan, Tainan",
        "cost_billion_usd": 3.1,
        "start_year": 2008,
        "wafer_size_mm": 300,
        "node_nm": 16,
        "capacity_wafers": 55000,
        "technology": "Foundry"
    },
    {
        "company": "GlobalFoundries",
        "facility": "Fab 1 Module 1",
        "location": "Germany, Saxony, Dresden",
        "cost_billion_usd": 3.6,
        "start_year": 2005,
        "wafer_size_mm": 300,
        "node_nm": 22,
        "capacity_wafers": 35000,
        "technology": "Foundry"
    }
]




# Extract costs, capacities, and process nodes from dictionary format
costs = [fab["cost_billion_usd"] for fab in fabs]
capacities = [fab["capacity_wafers"] for fab in fabs]
process_nodes = [fab["node_nm"] for fab in fabs]

# Create categories for process nodes
def categorize_node(node_nm):
    """Categorize process nodes into groups"""
    # node_nm is now an integer
    if node_nm <= 7:
        return "≤7nm (Leading Edge)"
    elif node_nm <= 14:
        return "10-14nm"
    elif node_nm <= 28:
        return "16-28nm"
    elif node_nm <= 65:
        return "32-65nm"
    else:
        return "≥90nm (Mature)"

# Categorize all nodes
node_categories = [categorize_node(node) for node in process_nodes]

# Define color mapping for node categories
node_colors = {
    "≤7nm (Leading Edge)": "#d62728",   # red
    "10-14nm": "#ff7f0e",               # orange
    "16-28nm": "#2ca02c",               # green
    "32-65nm": "#1f77b4",               # blue
    "≥90nm (Mature)": "#9467bd",        # purple
    "Other": "#8c564b"                  # brown
}

# Calculate regression line in linear space (forcing through origin: y = alpha * x)
# Use least squares to find alpha: alpha = sum(x*y) / sum(x^2)
alpha = np.sum(np.array(capacities) * np.array(costs)) / np.sum(np.array(capacities)**2)
x_range = np.linspace(0, max(capacities), 100)
y_range = alpha * x_range

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

# Plot points colored by process node category
for category in set(node_categories):
    mask = [c == category for c in node_categories]
    caps = [cap for cap, m in zip(capacities, mask) if m]
    cost = [c for c, m in zip(costs, mask) if m]
    ax.scatter(caps, cost, alpha=0.6, s=50,
               color=node_colors[category], label=category)

ax.set_xlabel('Fab Production Capacity (Wafers per Month)', fontsize=12)
ax.set_ylabel('Cost (Billions USD)', fontsize=12)

# Set specific tick marks for better readability (linear scale)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Add regression line with parameters
ax.plot(x_range, y_range, "--", alpha=0.5, color='gray', linewidth=2,
         label=f'Fit: y = {alpha:.6f}*x')

ax.legend(loc='best', fontsize=8)

plt.tight_layout()
plt.savefig('cost_of_fab_plot.png', dpi=300, bbox_inches='tight')

print(f"Plot saved as 'cost_of_fab_plot.png'")
print(f"\nCost of Fab:")
print(f"  Data points: {len(fabs)}")
print(f"  Cost range: ${min(costs):.3f}B - ${max(costs):.2f}B")
print(f"  Capacity range: {min(capacities):,} - {max(capacities):,} wafers/month")
