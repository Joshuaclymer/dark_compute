import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Read the capabilities.json file
with open('capabilities.json', 'r') as f:
    capabilities = json.load(f)

# Configuration
years = list(range(2005, 2026))  # 2005-2025
nodes = ['180nm', '90nm', '45nm', '28nm', '14nm', '7nm']
tool_types = ['DUV_scanner', 'Photoresist', 'Overlay_metrology']
tool_display_names = {
    'DUV_scanner': 'DUV Scanner',
    'Photoresist': 'Photoresist',
    'Overlay_metrology': 'Overlay Metrology'
}

# Use China data
china_data = capabilities['china']

def get_status(tool, node, year):
    """Determine if a tool is None, Pilot, or Production for a given node and year"""
    pilot_date = china_data.get('pilot', {}).get(node, {}).get(tool)
    hvm_date = china_data.get('hvm', {}).get(node, {}).get(tool)

    # Check if production (HVM) by this year
    if hvm_date is not None and year >= hvm_date:
        return 'Production'
    # Check if pilot by this year
    elif pilot_date is not None and year >= pilot_date:
        return 'Pilot'
    else:
        return 'None'

# Build the table data
table_data = []

for tool in tool_types:
    # Add a section header row
    section_header = {'Node': f'--- {tool_display_names[tool]} ---'}
    for year in years:
        section_header[str(year)] = ''
    table_data.append(section_header)

    # Add rows for each node in this section
    for node in nodes:
        row = {'Node': node}
        for year in years:
            status = get_status(tool, node, year)
            row[str(year)] = status
        table_data.append(row)

# Create DataFrame
df = pd.DataFrame(table_data)

# Print text version
print("\nChina's Semiconductor Tool Capability Timeline (2005-2025)")
print("DUV Scanner, Photoresist, and Overlay Metrology")
print("=" * 150)
print(df.to_string(index=False))
print("\n")

# Create color-coded visualization
fig, ax = plt.subplots(figsize=(20, 8))
ax.axis('tight')
ax.axis('off')

# Define colors for each capability level
color_map = {
    'Production': '#a8e6cf',  # Soft green
    'Pilot': '#ffd3b6',       # Soft orange
    'None': '#ffaaa5',        # Soft red
    '': '#e8e8e8'             # Gray for section headers
}

# Create cell colors
cell_colors = []
for i in range(len(df)):
    row_colors = []
    for j in range(len(df.columns)):
        if j == 0:  # Node column - use dark background
            row_colors.append('#34495e')
        else:
            value = df.iloc[i, j]
            row_colors.append(color_map.get(value, '#ffffff'))
    cell_colors.append(row_colors)

# Create the table
table = ax.table(cellText=df.values,
                colLabels=df.columns,
                cellColours=cell_colors,
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.8)

# Make the first column (Node) wider to accommodate section labels
for i in range(len(df) + 1):
    cell = table[(i, 0)]
    cell.set_width(0.15)  # Increase width from default

# Style header row
for i in range(len(df.columns)):
    cell = table[(0, i)]
    cell.set_facecolor('#34495e')
    cell.set_text_props(weight='bold', color='white')

# Style node column (first column) and section headers
for i in range(len(df)):
    cell = table[(i+1, 0)]
    cell.set_text_props(weight='bold', color='white')

    # Check if this is a section header row
    if '---' in str(df.iloc[i, 0]):
        # Make entire row bold and styled for section header
        for j in range(len(df.columns)):
            cell = table[(i+1, j)]
            cell.set_text_props(weight='bold', color='#34495e')
            cell.set_facecolor('#e8e8e8')

# Create legend
legend_elements = [
    mpatches.Patch(facecolor=color_map['Production'], label='Production (High Volume Manufacturing)'),
    mpatches.Patch(facecolor=color_map['Pilot'], label='Pilot (Limited Production)'),
    mpatches.Patch(facecolor=color_map['None'], label='None (No Capability)')
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=10)

plt.tight_layout()
plt.savefig('china_timeline_simple.png', dpi=300, bbox_inches='tight')
print("Color-coded timeline table saved to china_timeline_simple.png")
