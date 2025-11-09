import json
import matplotlib.pyplot as plt
import numpy as np
import sys

# Parse command-line argument for region
if len(sys.argv) > 1:
    region = sys.argv[1].lower()
else:
    region = 'both'

if region not in ['world', 'china', 'both']:
    print("Usage: python plot_capabilities.py [world|china|both]")
    print("Default: both")
    sys.exit(1)

# Load capabilities data from JSON
with open('capabilities.json', 'r') as f:
    capabilities_data = json.load(f)

# Define node names
nodes = ['180nm', '90nm', '45nm', '28nm', '14nm', '7nm']
node_labels = ['180 nm', '90 nm', '45 nm', '28 nm', '14 nm', '7 nm']

# Define tool names (matching JSON keys)
tool_names_map = {
    'DUV_scanner': 'DUV scanner',
    'Photomask_writer': 'Photomask writer',
    'Photoresist': 'Photoresist',
    'Overlay_metrology': 'Overlay metrology',
    'Ion_implanter': 'Ion implanter',
    'Electroplating': 'Electroplating',
    'deposition': 'Deposition',
    'etch': 'Etch'
}

# Order of tools for display (excluding Photomask_writer per original code)
tool_keys = ['DUV_scanner', 'Photoresist', 'Overlay_metrology', 'Ion_implanter', 'Electroplating', 'deposition', 'etch']
tool_names = [tool_names_map[key] for key in tool_keys]

# Node values for y-axis (in nm, for proper scaling)
node_values = [180, 90, 45, 28, 14, 7]

# Determine which regions to plot
if region == 'both':
    regions_to_plot = ['world', 'china']
else:
    regions_to_plot = [region]

# Create figure with extra space for legend
fig, ax = plt.subplots(figsize=(16, 8))

# Define colors for each tool type
colors = plt.cm.tab10(np.linspace(0, 1, len(tool_names)))

# Plot each region
for region_idx, current_region in enumerate(regions_to_plot):
    # Get data for current region from JSON
    region_data = capabilities_data[current_region]

    # For 'both' mode, use different line styles for each region
    if region == 'both':
        linestyle = '-' if current_region == 'world' else '--'
        alpha = 0.5
    else:
        linestyle = '-'
        alpha = 0.5

    # Plot each tool type
    for tool_idx, (tool_key, tool_name, color) in enumerate(zip(tool_keys, tool_names, colors)):
        # Collect data for each node
        pilot_points = []  # List of (year, node) tuples
        hvm_points = []    # List of (year, node) tuples

        for node_name, node_value in zip(nodes, node_values):
            # Get pilot year
            pilot_val = region_data['pilot'][node_name].get(tool_key)
            if pilot_val is not None:
                pilot_points.append((pilot_val, node_value))

            # Get HVM year
            hvm_val = region_data['hvm'][node_name].get(tool_key)
            if hvm_val is not None:
                hvm_points.append((hvm_val, node_value))

        # Don't modify the actual points - we'll handle backward compatibility
        # purely in the line drawing logic
        # Just keep the original points as-is

        # Filter out superseded points ONLY for display (not for line drawing)
        # We need to keep backward-compat points for the line, but hide them visually
        # For pilots: remove if superseded by a more advanced pilot OR HVM in the same year
        # For HVM: remove if superseded by a more advanced HVM in the same year

        # Separate: points to display vs points for line drawing
        display_pilot_data = []
        for year, node in pilot_points:
            # Check if superseded by more advanced pilot in same year
            superseded_by_pilot = any(y == year and n < node for y, n in pilot_points)
            # Check if superseded by HVM in same year
            superseded_by_hvm = any(y == year and n <= node for y, n in hvm_points)
            if not (superseded_by_pilot or superseded_by_hvm):
                display_pilot_data.append((year, node))

        display_hvm_data = []
        for year, node in hvm_points:
            # Check if superseded by more advanced HVM in same year
            superseded = any(y == year and n < node for y, n in hvm_points)
            if not superseded:
                display_hvm_data.append((year, node))

        # For line drawing, use ALL points (including backward compat)
        filtered_pilot_data = pilot_points
        filtered_hvm_data = hvm_points

        # Extract years and nodes for display (filtered to remove superseded)
        display_pilot_years = [y for y, n in display_pilot_data]
        display_pilot_nodes = [n for y, n in display_pilot_data]
        display_hvm_years = [y for y, n in display_hvm_data]
        display_hvm_nodes = [n for y, n in display_hvm_data]

        # Plot the line connecting points with the following rules:
        # (1) First point connects vertically from 180nm
        # (2) Lines only go down (to more advanced nodes), never up - skip points that would cause upward movement
        # (3) For each year, only plot the most advanced node achieved
        if filtered_pilot_data or filtered_hvm_data:
            # Combine all points
            all_points = [(y, n, 'hvm') for y, n in filtered_hvm_data]
            all_points += [(y, n, 'pilot') for y, n in filtered_pilot_data]

            # Group by year and find most advanced node per year
            from collections import defaultdict
            year_to_best_node = defaultdict(lambda: float('inf'))
            year_to_best_type = {}

            for year, node, ptype in all_points:
                if node < year_to_best_node[year]:
                    year_to_best_node[year] = node
                    year_to_best_type[year] = ptype
                elif node == year_to_best_node[year] and ptype == 'hvm':
                    # Prefer HVM over pilot if same node
                    year_to_best_type[year] = 'hvm'

            # Sort years chronologically
            sorted_years = sorted(year_to_best_node.keys())

            if sorted_years:
                # Track state
                first_year = sorted_years[0]
                first_node = year_to_best_node[first_year]

                # Rule (1): Draw vertical line from 180nm to first point
                ax.plot([first_year, first_year], [180, first_node],
                       linestyle, color=color, linewidth=2, alpha=alpha)

                prev_year = first_year
                prev_node = first_node

                # Process remaining years
                for year in sorted_years[1:]:
                    node = year_to_best_node[year]

                    # Rule (2): Only include points that don't go up (node must be <= prev_node to advance)
                    if node <= prev_node:
                        # Draw direct connection from previous point to current point
                        ax.plot([prev_year, year], [prev_node, node],
                               linestyle, color=color, linewidth=2, alpha=alpha)

                        prev_year = year
                        prev_node = node

                # Add dummy plot for legend (only for first region to avoid duplicate labels)
                if region_idx == 0:
                    # Pad tool name to match width of longest Status/Region label
                    padded_tool_name = tool_name.ljust(50)
                    ax.plot([], [], '-', color=color, linewidth=2, alpha=0.5, label=padded_tool_name)

            # Plot pilot points (hollow circles) with transparency - only display non-superseded
            if display_pilot_years:
                ax.scatter(display_pilot_years, display_pilot_nodes, s=120, facecolors='none',
                           edgecolors=color, linewidths=2.5, alpha=0.6, zorder=5)

            # Plot HVM points (solid circles) with transparency - only display non-superseded
            if display_hvm_years:
                ax.scatter(display_hvm_years, display_hvm_nodes, s=120, facecolors=color,
                           edgecolors=color, linewidths=2.5, alpha=0.6, zorder=5)

# Set y-axis to log scale for better visualization
ax.set_yscale('log')
ax.set_yticks(node_values)
ax.set_yticklabels(node_labels, fontsize="14")
ax.invert_yaxis()  # Smaller nodes at top (more advanced)
ax.set_ylabel("Process Node", fontsize=14, labelpad=10)
ax.set_xlabel("Year", fontsize=14, labelpad=10)

# Set labels and title
if region == 'both':
    title = 'World and China Semiconductor Manufacturing Capabilities Over Time'
else:
    title = f'{region.capitalize()} Semiconductor Manufacturing Capabilities Over Time'
ax.set_title(title, fontsize=16, pad=20)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--')

# Add legend for tools (primary legend)
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

tools_legend = ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1),
                         fontsize=12, framealpha=0.95, title='Tool Type',
                         title_fontsize=14, frameon=True, shadow=True,
                         handlelength=2, handleheight=1, borderpad=1.2, labelspacing=0.8)

# Add custom legend for point types and regions
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor='black', markersize=10, markeredgewidth=2,
           label='Passes verification', linestyle='None'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
           markeredgecolor='black', markersize=10, markeredgewidth=2,
           label='Ready for high\nyield manufacturing', linestyle='None')
]

# Add region line styles to legend if plotting both
if region == 'both':
    legend_elements.append(Line2D([0], [0], color='black', linewidth=2, linestyle='-', label='Tooling produced\nby any nation'))
    legend_elements.append(Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Tooling domestically produced in China'))

status_legend = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.04, 0.4),
                          fontsize=12, framealpha=0.95, title='Status/Region',
                          title_fontsize=14, frameon=True, shadow=True,
                          handlelength=2, handleheight=1, borderpad=1.3, labelspacing=0.8)

ax.add_artist(tools_legend)  # Keep both legends

# Make both legend boxes the same width
plt.draw()
tools_frame = tools_legend.get_frame()
status_frame = status_legend.get_frame()

# Get the renderer to calculate actual sizes
renderer = fig.canvas.get_renderer()
tools_bbox = tools_legend.get_window_extent(renderer)
status_bbox = status_legend.get_window_extent(renderer)

# Set both to the same width (use the wider one)
target_width = max(tools_bbox.width, status_bbox.width)
tools_frame.set_width(target_width / fig.dpi)
status_frame.set_width(target_width / fig.dpi)

# Set x-axis limits based on region
if region == 'world':
    ax.set_xlim(1982, 2024)
elif region == 'china':
    ax.set_xlim(2007, 2025)
else:  # both
    ax.set_xlim(1982, 2025)

# Set tick label font sizes
ax.tick_params(axis='x', labelsize=14)

# Adjust layout to prevent legend cutoff
plt.tight_layout()

# Save the plot with bbox_extra_artists to include legends
if region == 'both':
    output_file = 'world_and_china_semiconductor_capabilities.png'
else:
    output_file = f'{region}_semiconductor_capabilities.png'
plt.savefig(output_file, dpi=300,
            bbox_extra_artists=(tools_legend, status_legend),
            bbox_inches='tight')
print(f"Plot saved as '{output_file}'")

# Uncomment to show the plot interactively
# plt.show()
