"""
Example usage of the trajectory prediction module.

This demonstrates how to predict AI capability milestones from custom compute trajectories.
"""

import numpy as np
from predict_trajectory import (
    predict_milestones_from_compute,
    predict_trajectory_from_csv,
    print_milestone_summary,
    TrajectoryPredictor
)

# ============================================================================
# Example 1: Simple exponential growth
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 1: Exponential Compute Growth (Doubling every 2 years)")
print("="*80)

# Define time range
years = np.linspace(2024, 2045, 200)

# Exponential growth: doubling every 2 years
inference_compute = 1e6 * 2**((years - 2024) / 2.0)  # H100-years
experiment_compute = 1e5 * 2**((years - 2024) / 2.0)  # H100-years

# Predict milestones
milestones = predict_milestones_from_compute(
    time=years,
    inference_compute=inference_compute,
    experiment_compute=experiment_compute
)

# Print summary
print_milestone_summary(milestones)


# ============================================================================
# Example 2: Slower growth trajectory
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 2: Slower Growth (Doubling every 3 years)")
print("="*80)

# Slower exponential growth
inference_compute_slow = 1e6 * 2**((years - 2024) / 3.0)
experiment_compute_slow = 1e5 * 2**((years - 2024) / 3.0)

milestones_slow = predict_milestones_from_compute(
    time=years,
    inference_compute=inference_compute_slow,
    experiment_compute=experiment_compute_slow
)

print_milestone_summary(milestones_slow)


# ============================================================================
# Example 3: Compute growth slowdown
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 3: Compute Growth with Slowdown")
print("="*80)

# Fast growth initially, then slower
doubling_time = 2.0 + 0.2 * (years - 2024)  # Doubling time increases over time
cumulative_doublings = np.cumsum(1.0 / doubling_time) * (years[1] - years[0])
inference_compute_slowdown = 1e6 * 2**cumulative_doublings
experiment_compute_slowdown = 1e5 * 2**cumulative_doublings

milestones_slowdown = predict_milestones_from_compute(
    time=years,
    inference_compute=inference_compute_slowdown,
    experiment_compute=experiment_compute_slowdown
)

print_milestone_summary(milestones_slowdown)


# ============================================================================
# Example 4: Using TrajectoryPredictor for full trajectory access
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 4: Accessing Full Trajectory")
print("="*80)

predictor = TrajectoryPredictor()
milestones = predictor.predict_from_time_series(
    time=years,
    inference_compute=inference_compute,
    experiment_compute=experiment_compute
)

# Get full trajectory with all metrics
trajectory = predictor.get_full_trajectory()

print(f"Available trajectory metrics: {list(trajectory.keys())[:10]}... (and more)")
print(f"\nProgress trajectory shape: {trajectory['progress'].shape}")
print(f"Time range: {trajectory['times'][0]:.1f} to {trajectory['times'][-1]:.1f}")
print(f"Progress range: {trajectory['progress'][0]:.2f} to {trajectory['progress'][-1]:.2f} OOMs")

# Example: Get automation fraction over time
automation_fractions = trajectory['automation_fraction']
print(f"\nAutomation fraction:")
print(f"  Start (2024): {automation_fractions[0]:.1%}")
print(f"  End (2045): {automation_fractions[-1]:.1%}")


# ============================================================================
# Example 5: Load from CSV file
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 5: Loading from CSV (Default Input Data)")
print("="*80)

try:
    milestones_csv, trajectory_csv = predict_trajectory_from_csv('input_data.csv')
    print_milestone_summary(milestones_csv)

    # Show some trajectory info
    print(f"\nTrajectory time range: {trajectory_csv['times'][0]:.1f} to {trajectory_csv['times'][-1]:.1f}")
    print(f"Final progress: {trajectory_csv['progress'][-1]:.2f} OOMs")
except Exception as e:
    print(f"Could not load input_data.csv: {e}")


# ============================================================================
# Example 6: Compare different scenarios
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 6: Scenario Comparison")
print("="*80)

scenarios = {
    "Fast (1.5yr doubling)": 1.5,
    "Moderate (2yr doubling)": 2.0,
    "Slow (3yr doubling)": 3.0,
    "Very Slow (4yr doubling)": 4.0,
}

print(f"{'Scenario':<25} {'AC Time':<12} {'SAR Time':<12} {'ASI Time':<12}")
print("-" * 80)

for scenario_name, doubling_time in scenarios.items():
    inf_compute = 1e6 * 2**((years - 2024) / doubling_time)
    exp_compute = 1e5 * 2**((years - 2024) / doubling_time)

    ms = predict_milestones_from_compute(
        time=years,
        inference_compute=inf_compute,
        experiment_compute=exp_compute
    )

    ac_time = ms.get('AC').time if 'AC' in ms else None
    sar_time = ms.get('SAR-level-experiment-selection-skill').time if 'SAR-level-experiment-selection-skill' in ms else None
    asi_time = ms.get('ASI').time if 'ASI' in ms else None

    ac_str = f"{ac_time:.1f}" if ac_time else "N/A"
    sar_str = f"{sar_time:.1f}" if sar_time else "N/A"
    asi_str = f"{asi_time:.1f}" if asi_time else "N/A"

    print(f"{scenario_name:<25} {ac_str:<12} {sar_str:<12} {asi_str:<12}")

print("="*80 + "\n")
