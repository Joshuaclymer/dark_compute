import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np

# === CONFIGURATION PARAMETERS ===
# Adjust these to control spacing in the diagram
STAGE_TO_ARROW_GAP = 0.02  # Distance from bottom of stage to top of arrow
ARROW_LENGTH = 0.06  # Length of the arrow itself
ARROW_TO_TEXT_GAP = 0  # Distance from bottom of arrow to text label below
TEXT_TO_STAGE_GAP = 0.03  # Distance from text label to top of stage below
Y_START = 0.95  # Starting Y position for first stage

# Square parameters
SQUARE_SIZE = 0.008  # Size of each small square representing one machine
SQUARE_SPACING = 0.001  # Spacing between squares

# Data from lifecycle.py
total_machines = (2500, 3000)  # range
refurbished_annually = (125, 200)  # range
prop_cannibalized = (0, 0.05)  # range
prop_not_resold = (0, 0.1)  # range

# Calculate ranges and midpoints for each category
# Total manufactured
total_min, total_max = total_machines
total_mid = np.mean(total_machines)

# Refurbished (these are actual counts, not annual rates)
refurb_min, refurb_max = refurbished_annually
refurb_mid = np.mean(refurbished_annually)

# Cannibalized for parts (from refurbished machines)
cann_prop_min, cann_prop_max = prop_cannibalized
cann_min = refurb_min * cann_prop_min
cann_max = refurb_max * cann_prop_max
cann_mid = refurb_mid * np.mean(prop_cannibalized)

# Not resold (from cannibalized) - unaccounted for
not_resold_prop_min, not_resold_prop_max = prop_not_resold
unaccounted_min = cann_min * not_resold_prop_min
unaccounted_max = cann_max * not_resold_prop_max
unaccounted_mid = cann_mid * np.mean(prop_not_resold)

# Create funnel chart
fig, ax = plt.subplots(figsize=(14, 12))

# Define funnel stages (from top to bottom) - only the 4 from lifecycle.py
# Format: (label, midpoint, min, max)
stages = [
    ("Total Machines Manufactured\n(2000-present)", total_mid, total_min, total_max),
    ("Refurbished", refurb_mid, refurb_min, refurb_max),
    ("Cannibalized for Parts", cann_mid, cann_min, cann_max),
    ("Unaccounted For", unaccounted_mid, unaccounted_min, unaccounted_max),
]

# Calculate positions
y_start = Y_START
square_size = SQUARE_SIZE
square_spacing = SQUARE_SPACING

# Store box positions and text positions for arrow drawing
box_positions = []
text_positions = []

for i, (label, mid, low, high) in enumerate(stages):
    num_machines = int(round(mid))

    # Calculate how many squares per row to make a reasonable shape
    if i == 0:
        squares_per_row = 80  # Wide rectangle for top stage
    elif i == 1:
        squares_per_row = 20  # Narrower
    elif i == 2:
        squares_per_row = 3  # Small square-ish
    else:  # i == 3
        squares_per_row = 1  # Single column

    num_rows = int(np.ceil(num_machines / squares_per_row))

    # Calculate total width and height
    total_width = squares_per_row * (square_size + square_spacing)
    total_height = num_rows * (square_size + square_spacing)

    # Calculate y position based on previous elements
    if i == 0:
        y_top = y_start
    else:
        # Previous stage bottom + gap + arrow length + gap + text height estimate + gap
        prev_stage_bottom = box_positions[i - 1][0] - box_positions[i - 1][1] / 2
        text_height_estimate = 0.02  # Approximate height of text
        y_top = prev_stage_bottom - STAGE_TO_ARROW_GAP - ARROW_LENGTH - ARROW_TO_TEXT_GAP - text_height_estimate - TEXT_TO_STAGE_GAP

    y_center = y_top - total_height / 2
    x_start = (1 - total_width) / 2  # Center the grid
    y_bottom = y_center - total_height / 2

    # Store position for arrows
    box_positions.append((y_center, total_height))

    # Store text position (will be calculated properly later)
    if i > 0:
        text_y = y_top + TEXT_TO_STAGE_GAP
        text_positions.append(text_y)

    # Debug print
    print(f"Stage {i} ({label}): {num_machines} machines, {num_rows} rows x {squares_per_row} cols")

    # Draw individual squares for each machine
    machine_count = 0
    for row in range(num_rows):
        for col in range(squares_per_row):
            if machine_count >= num_machines:
                break

            x_pos = x_start + col * (square_size + square_spacing)
            y_pos = y_bottom + row * (square_size + square_spacing)

            square = Rectangle(
                (x_pos, y_pos),
                square_size,
                square_size,
                facecolor='#2d2d2d',
                edgecolor='white',
                linewidth=0.5,
                alpha=1.0
            )
            ax.add_patch(square)
            machine_count += 1

        if machine_count >= num_machines:
            break

    # Add text - format depends on stage
    # For stages 1-3 (refurbished, cannibalized, unaccounted), show range only centered above
    # For stage 0, don't add any label (already in title)
    if i > 0:
        # Stages 1-3: Show label and range centered above the rectangles
        if low < 1 and high < 1:
            range_text = f"{label}: {low:.1f} - {high:.1f} machines"
        elif high < 10:
            range_text = f"{label}: {int(low)} - {int(high)} machines"
        else:
            range_text = f"{label}: {int(low)} - {int(high)} machines"

        # Position text at the stored text position
        text_y_pos = text_positions[i - 1]
        ax.text(
            0.5, text_y_pos,
            range_text,
            ha='center',
            va='top',
            fontsize=13,
            fontweight='bold',
            color='black'
        )

# Add connecting arrows between stages

# Calculate proportions for each transition
# Stage 0 -> Stage 1: refurbished / total
prop_0_to_1_min = refurb_min / total_max  # Min proportion
prop_0_to_1_max = refurb_max / total_min  # Max proportion

# Stage 1 -> Stage 2: cannibalized / refurbished
prop_1_to_2_min = cann_prop_min
prop_1_to_2_max = cann_prop_max

# Stage 2 -> Stage 3: not resold / cannibalized
prop_2_to_3_min = not_resold_prop_min
prop_2_to_3_max = not_resold_prop_max

proportions = [
    (prop_0_to_1_min, prop_0_to_1_max),
    (prop_1_to_2_min, prop_1_to_2_max),
    (prop_2_to_3_min, prop_2_to_3_max),
]

for i in range(len(stages) - 1):
    y_center_top, box_h_top = box_positions[i]

    # Arrow starts below the stage with specified gap
    arrow_top = y_center_top - box_h_top / 2 - STAGE_TO_ARROW_GAP
    # Arrow ends after specified length
    arrow_bottom = arrow_top - ARROW_LENGTH

    # Draw arrow
    ax.annotate('',
                xy=(0.5, arrow_bottom),
                xytext=(0.5, arrow_top),
                arrowprops=dict(arrowstyle='->',
                              color='black',
                              lw=3,
                              alpha=0.7))

    # Add proportion label next to arrow
    prop_min, prop_max = proportions[i]
    arrow_y_mid = (arrow_top + arrow_bottom) / 2

    # Format proportion as percentage without brackets
    if prop_max < 0.01:
        # For very small proportions, show more decimal places
        prop_text = f"{prop_min*100:.2f}% - {prop_max*100:.2f}%"
    elif prop_max < 0.1:
        prop_text = f"{prop_min*100:.1f}% - {prop_max*100:.1f}%"
    else:
        prop_text = f"{prop_min*100:.1f}% - {prop_max*100:.1f}%"

    ax.text(
        0.52, arrow_y_mid,
        prop_text,
        ha='left',
        va='center',
        fontsize=13,
        fontweight='bold',
        color='black'
    )

# Title and styling
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.text(
    0.5, 0.99,
    'Photolithography Machines Produced by ASML for the 90 nm node and below: 2,500 - 3,000',
    ha='center',
    va='top',
    fontsize=13,
    fontweight='bold',
    color='black'
)

# No bottom note needed

plt.tight_layout()
plt.savefig('asml_lifecycle_funnel.png', dpi=300, bbox_inches='tight')
print("Funnel diagram saved as 'asml_lifecycle_funnel.png'")

# Also print summary statistics
print("\n=== ASML Photolithography Machine Lifecycle Summary ===")
print(f"Total Manufactured: {total_mid:.0f} machines [{total_min:.0f} - {total_max:.0f}]")
print(f"Refurbished: {refurb_mid:.0f} machines [{refurb_min:.0f} - {refurb_max:.0f}]")
print(f"Cannibalized for Parts: {cann_mid:.1f} machines [{cann_min:.1f} - {cann_max:.1f}]")
print(f"Unaccounted For: {unaccounted_mid:.2f} machines [{unaccounted_min:.2f} - {unaccounted_max:.2f}]")
