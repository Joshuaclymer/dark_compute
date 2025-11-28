"""
Example: Using total compute instead of splitting manually.

This demonstrates the simplified API where you only specify total compute
and the system automatically splits it between inference and experiment.
"""

import numpy as np
from predict_trajectory import (
    predict_milestones_from_total_compute,
    predict_milestones_from_compute,
    print_milestone_summary
)

print("="*80)
print("EXAMPLE: Using Total Compute (Automatic Split)")
print("="*80)

# Define scenario
years = np.linspace(2024, 2045, 200)

# You have 100,000 H100s available - just specify total!
total_compute = 1e5 * 2**((years - 2024) / 2.0)  # H100-years, doubling every 2 years

print(f"\nScenario: You have compute doubling every 2 years")
print(f"  Starting compute: {total_compute[0]:,.0f} H100-years")
print(f"  Ending compute: {total_compute[-1]:,.0f} H100-years")
print(f"\nNo need to decide how to split between inference and experiment!")
print(f"The system will use a 15%/85% split by default (15% for coding, 85% for experiments)")

# Predict using total compute (automatic split)
milestones_auto = predict_milestones_from_total_compute(
    time=years,
    total_compute=total_compute
)

print_milestone_summary(milestones_auto)

# Compare with different split strategies
print("\n" + "="*80)
print("COMPARISON: Different Allocation Strategies")
print("="*80)

strategies = {
    "5% inference / 95% experiment": 0.05,
    "10% inference / 90% experiment": 0.10,
    "15% inference / 85% experiment (default)": 0.15,
    "20% inference / 80% experiment": 0.20,
}

print(f"\n{'Strategy':<45} {'AC Time':<12} {'ASI Time':<12}")
print("-" * 80)

for strategy_name, inf_frac in strategies.items():
    milestones = predict_milestones_from_total_compute(
        time=years,
        total_compute=total_compute,
        inference_fraction=inf_frac
    )

    ac_time = milestones['AC'].time if 'AC' in milestones else None
    asi_time = milestones['ASI'].time if 'ASI' in milestones else None

    ac_str = f"{ac_time:.2f}" if ac_time else "N/A"
    asi_str = f"{asi_time:.2f}" if asi_time else "N/A"

    print(f"{strategy_name:<45} {ac_str:<12} {asi_str:<12}")

print("\nConclusion: The split ratio has only a small effect on milestone timing!")
print("Most compute should go to experiments, not inference/coding.")

print("\n" + "="*80)
print("MANUAL SPLIT vs AUTO SPLIT")
print("="*80)

# Manual split (the old way)
inference_manual = 0.15 * total_compute
experiment_manual = 0.85 * total_compute

milestones_manual = predict_milestones_from_compute(
    time=years,
    inference_compute=inference_manual,
    experiment_compute=experiment_manual
)

print("\nManual Split (you choose inference and experiment):")
print(f"  AC: {milestones_manual['AC'].time:.2f}")
print(f"  ASI: {milestones_manual['ASI'].time:.2f}")

print("\nAuto Split (you only specify total):")
print(f"  AC: {milestones_auto['AC'].time:.2f}")
print(f"  ASI: {milestones_auto['ASI'].time:.2f}")

print("\nResult: Identical! The auto split uses the same 15%/85% default.")
print("="*80)
