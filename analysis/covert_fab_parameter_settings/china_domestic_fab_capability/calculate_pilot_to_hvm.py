import json
import pandas as pd

# Read the capabilities.json file
with open('capabilities.json', 'r') as f:
    capabilities = json.load(f)

# Tool display names
tool_display_names = {
    'DUV_scanner': 'DUV Scanner',
    'Photomask_writer': 'Photomask Writer',
    'Photoresist': 'Photoresist',
    'Overlay_metrology': 'Overlay Metrology',
    'Ion_implanter': 'Ion Implanter',
    'Electroplating': 'Electroplating',
    'deposition': 'Deposition',
    'etch': 'Etch'
}

def find_transitions(region_data, region_name):
    """Find pilot->HVM transitions for a region"""
    transitions = []

    pilot_data = region_data.get('pilot', {})
    hvm_data = region_data.get('hvm', {})

    # Get all nodes
    all_nodes = set(pilot_data.keys()) | set(hvm_data.keys())

    for node in all_nodes:
        pilot_tools = pilot_data.get(node, {})
        hvm_tools = hvm_data.get(node, {})

        # Get all tool types for this node
        all_tools = set(pilot_tools.keys()) | set(hvm_tools.keys())

        for tool in all_tools:
            pilot_date = pilot_tools.get(tool)
            hvm_date = hvm_tools.get(tool)

            # Only include if both pilot and HVM dates exist
            if pilot_date is not None and hvm_date is not None:
                time_diff = hvm_date - pilot_date

                # Only include if HVM came at or after pilot
                if time_diff >= 0:
                    tool_name = tool_display_names.get(tool, tool)
                    node_nm = int(node.replace('nm', ''))

                    transitions.append({
                        'Region': region_name,
                        'Node (nm)': node_nm,
                        'Tool Type': tool_name,
                        'Pilot Date': pilot_date,
                        'HVM Date': hvm_date,
                        'Time Difference (years)': round(time_diff, 1)
                    })

    return transitions

# Find transitions for both regions
china_transitions = find_transitions(capabilities['china'], 'China')
world_transitions = find_transitions(capabilities['world'], 'World')

# Combine and sort
all_transitions = china_transitions + world_transitions
all_transitions_df = pd.DataFrame(all_transitions)

# Sort by region, node size (descending), and tool type
all_transitions_df = all_transitions_df.sort_values(['Region', 'Node (nm)', 'Tool Type'], ascending=[True, False, True])

print("# Pilot to HVM Transition Times\n")
print(all_transitions_df.to_markdown(index=False))
print("\n")

# Also create separate tables
if china_transitions:
    china_df = pd.DataFrame(china_transitions).sort_values(['Node (nm)', 'Tool Type'], ascending=[False, True])
    print("## China Only\n")
    print(china_df.to_markdown(index=False))
    print("\n")

if world_transitions:
    world_df = pd.DataFrame(world_transitions).sort_values(['Node (nm)', 'Tool Type'], ascending=[False, True])
    print("## World Only\n")
    print(world_df.to_markdown(index=False))
