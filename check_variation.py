#!/usr/bin/env python3
"""Check if simulations are producing varied results"""
from model import Model
import numpy as np

print("Running 100 simulations to check variation...")
model = Model(year_us_prc_agreement_goes_into_force=2028, end_year=2035, increment=0.1)
model.run_simulations(num_simulations=100)

# Extract data to analyze variation
us_probs_by_sim = []
h100e_by_sim = []

for covert_projects, detectors in model.simulation_results:
    us_beliefs = detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"]
    h100e_over_time = covert_projects["prc_covert_project"].h100e_over_time

    years = sorted(us_beliefs.keys())
    us_probs = [us_beliefs[year].p_covert_fab_exists for year in years]
    h100e_counts = [h100e_over_time.get(year, 0.0) for year in years]

    us_probs_by_sim.append(us_probs)
    h100e_by_sim.append(h100e_counts)

us_probs_array = np.array(us_probs_by_sim)
h100e_array = np.array(h100e_by_sim)

print("\n=== US PROBABILITY VARIATION ===")
print(f"Years analyzed: {len(years)}")
for i, year in enumerate(years[::10]):  # Sample every 10th year
    values = us_probs_array[:, i * 10] if i * 10 < us_probs_array.shape[1] else us_probs_array[:, -1]
    print(f"Year {year:.1f}:")
    print(f"  Min: {values.min():.4f}, Max: {values.max():.4f}, Std: {values.std():.4f}")
    print(f"  Median: {np.median(values):.4f}")
    print(f"  Sample values: {values[:5]}")

print("\n=== H100e COUNT VARIATION ===")
for i, year in enumerate(years[::10]):
    values = h100e_array[:, i * 10] if i * 10 < h100e_array.shape[1] else h100e_array[:, -1]
    print(f"Year {year:.1f}:")
    print(f"  Min: {values.min():.1f}, Max: {values.max():.1f}, Std: {values.std():.1f}")
    print(f"  Median: {np.median(values):.1f}")

print("\n=== CHECKING IF ALL SIMULATIONS ARE IDENTICAL ===")
# Check if all probability trajectories are the same
all_identical = True
first_sim = us_probs_array[0]
for i in range(1, len(us_probs_array)):
    if not np.allclose(us_probs_array[i], first_sim, atol=1e-10):
        all_identical = False
        break

if all_identical:
    print("⚠️  WARNING: All simulations produced IDENTICAL US probability trajectories!")
    print("This explains why no individual traces are visible.")
else:
    print("✓ Simulations have variation")

print("\n=== SAMPLE INDIVIDUAL TRAJECTORIES ===")
print("First 3 simulations at year 2030:")
year_2030_idx = years.index(2030.0) if 2030.0 in years else len(years) // 2
for i in range(min(3, len(us_probs_array))):
    print(f"  Sim {i+1}: P={us_probs_array[i, year_2030_idx]:.6f}, H100e={h100e_array[i, year_2030_idx]:.1f}")
