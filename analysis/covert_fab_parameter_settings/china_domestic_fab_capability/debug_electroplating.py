import csv

def read_csv_with_variable_cols(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        rows = []
        for row in reader:
            rows.append(row)
    return rows

world_pilot = read_csv_with_variable_cols('world_pilot.csv')
world_hvm = read_csv_with_variable_cols('world_hvm.csv')

nodes = ['180 nm', '90 nm', '45 nm', '28 nm', '14 nm', '7 nm']
node_values = [180, 90, 45, 28, 14, 7]

# Electroplating is row index 6 (0-indexed, skipping header)
row_idx = 6

pilot_data = []
hvm_data = []

for node_idx, (node_name, node_value) in enumerate(zip(nodes, node_values)):
    col_idx = node_idx + 1

    if row_idx < len(world_pilot) and col_idx < len(world_pilot[row_idx]):
        pilot_val = world_pilot[row_idx][col_idx]
        if pilot_val and pilot_val.strip():
            try:
                pilot_data.append((float(pilot_val), node_value))
                print(f"Pilot: {node_name} at {pilot_val}")
            except ValueError:
                pass

    if row_idx < len(world_hvm) and col_idx < len(world_hvm[row_idx]):
        hvm_val = world_hvm[row_idx][col_idx]
        if hvm_val and hvm_val.strip():
            try:
                hvm_data.append((float(hvm_val), node_value))
                print(f"HVM: {node_name} at {hvm_val}")
            except ValueError:
                pass

print("\nBefore filtering:")
print(f"Pilot data: {pilot_data}")
print(f"HVM data: {hvm_data}")

# Filter out superseded points
filtered_pilot_data = []
for year, node in pilot_data:
    superseded_by_pilot = any(y == year and n < node for y, n in pilot_data)
    superseded_by_hvm = any(y == year and n <= node for y, n in hvm_data)
    if not (superseded_by_pilot or superseded_by_hvm):
        filtered_pilot_data.append((year, node))
    else:
        print(f"Filtered out pilot: year={year}, node={node}nm (superseded_by_pilot={superseded_by_pilot}, superseded_by_hvm={superseded_by_hvm})")

filtered_hvm_data = []
for year, node in hvm_data:
    superseded = any(y == year and n < node for y, n in hvm_data)
    if not superseded:
        filtered_hvm_data.append((year, node))
    else:
        print(f"Filtered out HVM: year={year}, node={node}nm")

print("\nAfter filtering:")
print(f"Filtered pilot data: {filtered_pilot_data}")
print(f"Filtered HVM data: {filtered_hvm_data}")
