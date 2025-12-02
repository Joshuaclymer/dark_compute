"""
Test script to predict AI takeoff trajectories using the takeoff model.

This script:
1. Loads example compute time series from our model (global and covert)
2. Converts our compute units (H100e TPP) to takeoff model units (H100-years)
3. Runs trajectory predictions using the takeoff model
4. Prints key milestone times (especially AAR to ASI)
"""

import sys
import os
import numpy as np

# Change to takeoff model directory (required for the model to load properly)
takeoff_model_dir = '/Users/joshuaclymer/github/covert_compute_production_model/takeoff_model/ai-futures-calculator'
os.chdir(takeoff_model_dir)
sys.path.insert(0, takeoff_model_dir)

from predict_trajectory import predict_milestones_from_total_compute, print_milestone_summary

# Constants from our model
H100_TPP_PER_CHIP = 2144.0  # Tera-Parameter-Passes per H100 chip
SECONDS_PER_YEAR = 365.25 * 24 * 3600  # ~31,557,600 seconds
H100_FLOPS = 1.979e15  # H100 FP16 peak FLOPS (from 989 TFLOPS Tensor Core * 2 for FP16)

def h100e_tpp_to_h100_years(h100e_tpp_stock, utilization=0.5):
    """
    Convert H100e TPP (stock of compute) to H100-years.

    Args:
        h100e_tpp_stock: Stock of H100-equivalent chips (in H100e TPP units)
        utilization: Fraction of time chips are running (default 0.5 = 50%)

    Returns:
        Equivalent H100-years of compute

    The conversion logic:
    - h100e_tpp_stock represents compute capacity in units of H100 equivalents
    - To get H100-years, we just multiply by utilization
    - 1 H100e TPP running at 100% for 1 year = 1 H100-year
    """
    return h100e_tpp_stock * utilization


def create_example_compute_trajectories():
    """
    Create example compute trajectories for testing.

    Returns:
        dict with 'years', 'global_compute', 'covert_compute' (all in H100-years)
    """
    # Time range: 2024 to 2040
    years = np.linspace(2024, 2040, 160)  # Quarterly resolution

    # Example 1: Global compute (no slowdown scenario)
    # Assume exponential growth: 3x per year (very aggressive)
    global_h100e_tpp_2024 = 1e6  # 1 million H100e equivalent stock in 2024
    growth_rate_per_year = 3.0
    global_h100e_tpp = global_h100e_tpp_2024 * (growth_rate_per_year ** (years - 2024))

    # Convert to H100-years (assume 50% utilization)
    global_compute_h100_years = h100e_tpp_to_h100_years(global_h100e_tpp, utilization=0.5)

    # Example 2: Covert compute (with slowdown)
    # Assume slowdown starts in 2030, much slower growth after
    covert_h100e_tpp = np.zeros_like(years)
    for i, year in enumerate(years):
        if year < 2030:
            # Before slowdown: same as global
            covert_h100e_tpp[i] = global_h100e_tpp_2024 * (growth_rate_per_year ** (year - 2024))
        else:
            # After slowdown: much slower growth (1.2x per year)
            # Start from 2030 value and grow slowly
            val_2030 = global_h100e_tpp_2024 * (growth_rate_per_year ** (2030 - 2024))
            covert_h100e_tpp[i] = val_2030 * (1.2 ** (year - 2030))

    # Convert to H100-years
    covert_compute_h100_years = h100e_tpp_to_h100_years(covert_h100e_tpp, utilization=0.5)

    return {
        'years': years,
        'global_compute': global_compute_h100_years,
        'covert_compute': covert_compute_h100_years
    }


def predict_and_compare_trajectories():
    """
    Predict trajectories for both global and covert compute scenarios.
    Print comparison of key milestones.
    """
    print("="*80)
    print("AI TAKEOFF TRAJECTORY PREDICTION TEST")
    print("="*80)
    print()

    # Start with the exact same setup as example_total_compute.py
    years = np.linspace(2024, 2045, 200)
    total_compute = 1e5 * 2**((years - 2024) / 2.0)  # H100-years, doubling every 2 years

    print("TEST 1: Replicating example_total_compute.py")
    print(f"Time range: {years[0]:.1f} to {years[-1]:.1f}")
    print(f"Starting compute: {total_compute[0]:,.0f} H100-years")
    print(f"Ending compute: {total_compute[-1]:,.0f} H100-years")
    print()

    try:
        milestones = predict_milestones_from_total_compute(
            time=years,
            total_compute=total_compute
        )
        print("SUCCESS! Got milestones:")
        for name, info in sorted(milestones.items(), key=lambda x: x[1].time):
            print(f"  {name}: {info.time:.2f} years")

        # Now try our custom scenarios
        print("\n" + "="*80)
        print("TEST 2: Custom scenarios")
        print("="*80)

        # Create example compute trajectories
        data = create_example_compute_trajectories()
        years_custom = data['years']
        global_compute = data['global_compute']
        covert_compute = data['covert_compute']

        print("\nCOMPUTE TRAJECTORIES:")
        print(f"Time range: {years_custom[0]:.1f} to {years_custom[-1]:.1f}")
        print(f"Global compute in 2024: {global_compute[0]:.2e} H100-years")
        print(f"Global compute in 2040: {global_compute[-1]:.2e} H100-years")
        print(f"Covert compute in 2024: {covert_compute[0]:.2e} H100-years")
        print(f"Covert compute in 2030: {covert_compute[np.argmin(np.abs(years_custom - 2030))]:.2e} H100-years")
        print(f"Covert compute in 2040: {covert_compute[-1]:.2e} H100-years")
        print()

        # Predict milestones for global scenario
        print("-"*80)
        print("SCENARIO 1: GLOBAL COMPUTE (NO SLOWDOWN)")
        print("-"*80)
        global_milestones = predict_milestones_from_total_compute(
            time=years_custom,
            total_compute=global_compute
        )

        ac_time_global = global_milestones.get('AC', None)
        asi_time_global = global_milestones.get('ASI', None)

        if ac_time_global:
            print(f"AC: {ac_time_global.time:.2f} years")
        if asi_time_global:
            print(f"ASI: {asi_time_global.time:.2f} years")
        if ac_time_global and asi_time_global:
            time_to_asi_global = asi_time_global.time - ac_time_global.time
            print(f"Time from AC to ASI: {time_to_asi_global:.2f} years")

        print()

        # Predict milestones for covert scenario
        print("-"*80)
        print("SCENARIO 2: COVERT COMPUTE (WITH SLOWDOWN)")
        print("-"*80)
        covert_milestones = predict_milestones_from_total_compute(
            time=years_custom,
            total_compute=covert_compute
        )

        ac_time_covert = covert_milestones.get('AC', None)
        asi_time_covert = covert_milestones.get('ASI', None)

        if ac_time_covert:
            print(f"AC: {ac_time_covert.time:.2f} years")
        if asi_time_covert:
            print(f"ASI: {asi_time_covert.time:.2f} years")
        if ac_time_covert and asi_time_covert:
            time_to_asi_covert = asi_time_covert.time - ac_time_covert.time
            print(f"Time from AC to ASI: {time_to_asi_covert:.2f} years")

        print()

        # Compare scenarios
        print("="*80)
        print("COMPARISON")
        print("="*80)

        if ac_time_global and ac_time_covert:
            ac_delay = ac_time_covert.time - ac_time_global.time
            print(f"AC delay (covert vs global): {ac_delay:.2f} years")
            print(f"  Global AC: {ac_time_global.time:.2f}")
            print(f"  Covert AC: {ac_time_covert.time:.2f}")

        if asi_time_global and asi_time_covert:
            asi_delay = asi_time_covert.time - asi_time_global.time
            print(f"\nASI delay (covert vs global): {asi_delay:.2f} years")
            print(f"  Global ASI: {asi_time_global.time:.2f}")
            print(f"  Covert ASI: {asi_time_covert.time:.2f}")

        if (ac_time_global and asi_time_global and
            ac_time_covert and asi_time_covert):
            time_to_asi_global = asi_time_global.time - ac_time_global.time
            time_to_asi_covert = asi_time_covert.time - ac_time_covert.time
            slowdown_factor = time_to_asi_covert / time_to_asi_global if time_to_asi_global > 0 else float('inf')

            print(f"\nTime from AC to ASI:")
            print(f"  Global: {time_to_asi_global:.2f} years")
            print(f"  Covert: {time_to_asi_covert:.2f} years")
            print(f"  Slowdown factor: {slowdown_factor:.2f}x")

        print("="*80)

    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    predict_and_compare_trajectories()
