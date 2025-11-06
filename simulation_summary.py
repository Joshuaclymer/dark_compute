"""
Simulate covert fab detection and create a CCDF of H100 equivalents produced before detection.

This script runs simulations for different agreement start years (2027 and 2030) and plots
the complementary cumulative distribution function (CCDF) showing P(H100e > x) before detection.
"""

import numpy as np
import matplotlib.pyplot as plt
from model import Model

# Configuration
AGREEMENT_START_YEARS = [2027, 2030]
NUM_SIMULATIONS = 10000
DETECTION_THRESHOLD = 0.5  # 50% probability threshold for detection


def get_h100e_at_detection(model):
    """
    Extract H100e produced at detection for each simulation.

    Args:
        model: Model instance with simulation results

    Returns:
        List of H100e values at detection (one per simulation)
    """
    h100e_values = []

    for covert_projects, detectors in model.simulation_results:
        prc_covert_project = covert_projects['prc_covert_project']
        if prc_covert_project and prc_covert_project.covert_fab:
            # Get US intelligence beliefs and H100e production
            us_beliefs = detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"]
            h100e_over_time = prc_covert_project.h100e_over_time

            # Find first year when US probability exceeds threshold
            years = sorted(us_beliefs.keys())
            detection_year = None
            for year in years:
                if us_beliefs[year].p_covert_fab_exists >= DETECTION_THRESHOLD:
                    detection_year = year
                    break

            # Include all simulations with covert fabs (detected or not)
            if detection_year is not None:
                h100e_at_detection = h100e_over_time.get(detection_year, 0.0)
                h100e_values.append(h100e_at_detection)
            else:
                # If never detected, use final H100e value
                final_year = max(h100e_over_time.keys()) if h100e_over_time else None
                if final_year:
                    h100e_values.append(h100e_over_time[final_year])
                else:
                    h100e_values.append(0.0)

    return h100e_values


def compute_ccdf(values):
    """
    Compute the complementary cumulative distribution function.

    Args:
        values: List of numeric values

    Returns:
        x_values: Sorted values
        ccdf_values: P(X >= x) for each x
    """
    if not values:
        return [], []

    # Sort values
    sorted_values = np.sort(values)

    # Compute CCDF: P(X >= x) using same formula as original
    ccdf_values = 1.0 - np.arange(1, len(sorted_values) + 1) / len(sorted_values)

    return sorted_values, ccdf_values


def run_analysis():
    """Run simulations for different agreement start years and compute CCDFs."""
    results = {}

    for year in AGREEMENT_START_YEARS:
        print(f"Running {NUM_SIMULATIONS} simulations for agreement start year {year}...")

        # Create model with this agreement start year
        model = Model(
            year_us_prc_agreement_goes_into_force=year,
            end_year=2037,
            increment=0.1
        )

        # Run simulations
        model.run_simulations(num_simulations=NUM_SIMULATIONS)

        # Get H100e values at detection
        h100e_values = get_h100e_at_detection(model)
        print(f"  Collected {len(h100e_values)} H100e values")

        # Compute CCDF
        x_values, ccdf_values = compute_ccdf(h100e_values)
        results[year] = (x_values, ccdf_values)

        # Print summary statistics
        if h100e_values:
            print(f"  Mean H100e: {np.mean(h100e_values):,.0f}")
            print(f"  Median H100e: {np.median(h100e_values):,.0f}")
            print(f"  Max H100e: {np.max(h100e_values):,.0f}")

    return results


def plot_results(results, save_path='simulation_summary.png'):
    """Plot CCDF for each agreement start year."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Different colors for each year
    colors = ['#FF8C00', '#1E90FF']  # Orange and blue

    for i, year in enumerate(AGREEMENT_START_YEARS):
        x_values, ccdf_values = results[year]
        # Convert CCDF values to percentages
        ccdf_percentages = [p * 100 for p in ccdf_values]
        ax.plot(x_values, ccdf_percentages,
                linewidth=2, color=colors[i],
                label=f'Agreement starts {year}')

    # Add markers at 100K and 1M H100e for 2030 line only
    target_computes = [100000, 1000000]  # 100K and 1M H100e
    year_2030_idx = AGREEMENT_START_YEARS.index(2030)
    x_values_2030, ccdf_values_2030 = results[2030]

    for target_compute in target_computes:
        if len(x_values_2030) > 0 and x_values_2030[-1] > target_compute:
            # Find the CCDF value at target H100e
            idx = np.searchsorted(x_values_2030, target_compute)
            ccdf_at_target = ccdf_values_2030[idx] if idx < len(ccdf_values_2030) else ccdf_values_2030[-1]
            ccdf_percentage = ccdf_at_target * 100

            # Add a marker point with label
            ax.plot(target_compute, ccdf_percentage, 'o', color='black', markersize=8, zorder=10)

            # Add text label showing the percentage
            ax.text(target_compute, ccdf_percentage, f'  {ccdf_percentage:.1f}%',
                   fontsize=9, verticalalignment='center', horizontalalignment='left')

    ax.set_xlabel('Total H100 equivalents produced before detection', fontsize=12)
    ax.set_ylabel('P(Compute Produced ≥ x)', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Use log scale for x-axis
    ax.set_xscale('log')

    # Format x-axis ticks with clear labels
    from matplotlib.ticker import PercentFormatter, FuncFormatter
    def format_thousands(x, pos):
        """Format large numbers with K, M suffixes"""
        if x >= 1e6:
            return f'{x/1e6:.0f}M'
        elif x >= 1e3:
            return f'{x/1e3:.0f}K'
        else:
            return f'{x:.0f}'

    ax.xaxis.set_major_formatter(FuncFormatter(format_thousands))

    # Format y-axis ticks as percentages
    ax.yaxis.set_major_formatter(PercentFormatter())

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    print("Starting simulation summary analysis...")
    print(f"Agreement years: {AGREEMENT_START_YEARS}")
    print(f"Simulations per year: {NUM_SIMULATIONS}\n")

    results = run_analysis()
    plot_results(results)

    print("\nAnalysis complete!")
