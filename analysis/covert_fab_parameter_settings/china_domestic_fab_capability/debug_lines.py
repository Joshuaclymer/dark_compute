import csv

def read_csv_with_variable_cols(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        rows = []
        for row in reader:
            rows.append(row)
    return rows

china_pilot = read_csv_with_variable_cols('china_pilot.csv')
china_hvm = read_csv_with_variable_cols('china_hvm.csv')

nodes = ['180 nm', '90 nm', '45 nm', '28 nm', '14 nm', '7 nm']
node_values = [180, 90, 45, 28, 14, 7]

# DUV scanner is row 1
row_idx = 1

pilot_points = []
for node_idx, (node_name, node_value) in enumerate(zip(nodes, node_values)):
    col_idx = node_idx + 1
    if row_idx < len(china_pilot) and col_idx < len(china_pilot[row_idx]):
        pilot_val = china_pilot[row_idx][col_idx]
        if pilot_val and pilot_val.strip():
            try:
                pilot_points.append((float(pilot_val), node_value))
            except ValueError:
                pass

print("Original pilot points:", pilot_points)

# Apply backward compatibility
pilot_points_sorted = sorted(pilot_points, key=lambda x: x[0])
new_pilot_points = []
pilot_nodes_achieved = {}

for year, node in pilot_points_sorted:
    if node not in pilot_nodes_achieved:
        pilot_nodes_achieved[node] = year
        new_pilot_points.append((year, node))
        print(f"Adding pilot: {year}, {node}nm")

    for larger_node in node_values:
        if larger_node > node:
            if larger_node not in pilot_nodes_achieved:
                pilot_nodes_achieved[larger_node] = year
                new_pilot_points.append((year, larger_node))
                print(f"  Backward compat: {year}, {larger_node}nm")

print("\nAfter backward compatibility:", new_pilot_points)

# Now simulate filtering
filtered_pilot_data = []
for year, node in new_pilot_points:
    superseded_by_pilot = any(y == year and n < node for y, n in new_pilot_points)
    if not superseded_by_pilot:
        filtered_pilot_data.append((year, node))
    else:
        print(f"Filtered out: {year}, {node}nm")

print("\nAfter filtering:", filtered_pilot_data)

# Now simulate line building
all_points = [(y, n, 'pilot') for y, n in filtered_pilot_data]
all_points.sort(key=lambda x: (x[0], 0 if x[2] == 'hvm' else 1, x[1]))

line_years = []
line_nodes = []
current_best_node = float('inf')

for year, node, ptype in all_points:
    if ptype == 'hvm' or node < current_best_node:
        line_years.append(year)
        line_nodes.append(node)
        current_best_node = min(current_best_node, node)
        print(f"Line point: {year}, {node}nm")

print("\nFinal line points:", list(zip(line_years, line_nodes)))

# Check for vertical segments
print("\nVertical segments:")
for i in range(len(line_years)):
    if i > 0:
        prev_year = line_years[i-1]
        prev_node = line_nodes[i-1]
        curr_year = line_years[i]
        curr_node = line_nodes[i]

        if curr_year == prev_year and curr_node != prev_node:
            print(f"VERTICAL: year={curr_year}, from {prev_node}nm to {curr_node}nm")
        else:
            print(f"Horizontal: year {prev_year} to {curr_year}, node {prev_node}nm to {curr_node}nm")
