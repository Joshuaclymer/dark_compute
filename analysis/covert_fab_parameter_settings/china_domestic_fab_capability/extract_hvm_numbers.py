import csv
import re
import sys
import os

def extract_hvm_year(cell_value):
    """
    Extract the HVM year from a cell.
    Returns the year, average of range, or empty string for N/A.

    Handles two formats:
    1. China format: "YYYY (pilot); YYYY (HVM); ..."
    2. World format: "YYYY; YYYY; ..." where second date is HVM
    """
    if not cell_value or cell_value.strip() == '':
        return ''

    # First try the China format with "(HVM)" label
    hvm_pattern = r'(\d{4})(?:-(\d{4}))?\s*\(HVM\)'
    match = re.search(hvm_pattern, cell_value)

    if match:
        start_year = int(match.group(1))
        end_year = match.group(2)

        if end_year:
            # It's a range, calculate average
            end_year = int(end_year)
            avg_year = (start_year + end_year) / 2
            return str(avg_year)
        else:
            # Single year
            return str(start_year)

    # Try world format: second date after first semicolon
    # Pattern: skip first date, find second year or year-range
    world_pattern = r'^\d{4}(?:-\d{4})?\s*;\s*(\d{4})(?:-(\d{4}))?\s*;'
    match = re.search(world_pattern, cell_value)

    if match:
        start_year = int(match.group(1))
        end_year = match.group(2)

        if end_year:
            # It's a range, calculate average
            end_year = int(end_year)
            avg_year = (start_year + end_year) / 2
            return str(avg_year)
        else:
            # Single year
            return str(start_year)

    # Check if it's explicitly N/A or NA
    if 'N/A (HVM)' in cell_value or cell_value.strip().upper().startswith('N/A') or cell_value.strip().upper() == 'NA':
        return ''

    # If no HVM pattern found, return empty
    return ''

def main():
    # Check if input file is provided as argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'china.csv'

    # Generate output filename by replacing .csv with _hvm.csv
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_hvm.csv"

    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)

        with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile)

            for row in reader:
                # Extract HVM year from each cell in the row
                hvm_row = [extract_hvm_year(cell) for cell in row]
                writer.writerow(hvm_row)

    print(f"Successfully created {output_file}")
    print("HVM years extracted (ranges averaged, N/A left blank)")

if __name__ == '__main__':
    main()
