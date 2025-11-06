"""
Analyze how the probability of achieving compute thresholds varies with agreement start year.

This script runs simulations for different agreement start years (2026-2030) and plots
the probability of producing >100K, >500K, and >2M H100e before detection.
"""

import numpy as np
import matplotlib.pyplot as plt
from model import Model

# Configuration
AGREEMENT_START_YEARS = range(2026, 2031)  # 2026 to 2030
COMPUTE_THRESHOLDS = [100000, 500000, 2000000]  # 100K, 500K, 2M H100e
THRESHOLD_LABELS = ['P(>100K H100e before detection)', 'P(>500K H100e before detection)', 'P(>2M H100e before detection)']
NUM_SIMULATIONS = 10000
DETECTION_THRESHOLD = 0.5  # 50% probability threshold for detection

def calculate_success_probability(model, threshold):
    """
    Calculate the probability that a covert fab produces > threshold H100e before detection.

    Args:
        model: Model instance with simulation results
        threshold: H100e threshold to check

    Returns:
        Probability (0 to 1) of achieving the threshold before detection
    """
    if not model.simulation_results:
        return 0.0

    success_count = 0
    total_count = 0

    for covert_projects, detectors in model.simulation_results:
        prc_covert_project = covert_projects['prc_covert_project']
        if prc_covert_project and prc_covert_project.covert_fab:
            total_count += 1

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

            # Check if this simulation achieved threshold before detection
            if detection_year is not None:
                h100e_at_detection = h100e_over_time.get(detection_year, 0.0)
                if h100e_at_detection >= threshold:
                    success_count += 1

    if total_count == 0:
        return 0.0

    return success_count / total_count


def run_analysis():
    """Run simulations for different agreement start years and compute success probabilities."""
    # Store results: threshold -> list of probabilities (one per agreement year)
    results = {threshold: [] for threshold in COMPUTE_THRESHOLDS}

    for year in AGREEMENT_START_YEARS:
        print(f"Running simulations for agreement start year {year}...")

        # Create model with this agreement start year
        model = Model(
            year_us_prc_agreement_goes_into_force=year,
            end_year=2037,
            increment=0.1
        )

        # Run simulations
        model.run_simulations(num_simulations=NUM_SIMULATIONS)

        # Calculate success probability for each threshold
        for threshold in COMPUTE_THRESHOLDS:
            prob = calculate_success_probability(model, threshold)
            results[threshold].append(prob)
            print(f"  P(>{threshold:,} H100e) = {prob:.3f}")

    return results


def plot_results(results, save_path='agreement_year_analysis.png'):
    """Plot the results showing probability vs agreement start year for each threshold."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Plot a line for each threshold with consistent color but different markers
    orange_color = '#FF8C00'  # Orange color
    markers = ['o', 's', '^']  # Circle, square, triangle
    for i, (threshold, label) in enumerate(zip(COMPUTE_THRESHOLDS, THRESHOLD_LABELS)):
        probabilities = [p * 100 for p in results[threshold]]  # Convert to percentages
        ax.plot(AGREEMENT_START_YEARS, probabilities,
                marker=markers[i], linewidth=2, markersize=8,
                color=orange_color, label=label)

    ax.set_xlabel('Agreement Start Year', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(list(AGREEMENT_START_YEARS))
    ax.set_ylim([0, max([max(results[t]) for t in COMPUTE_THRESHOLDS]) * 110])  # 10% headroom

    # Format y-axis ticks as percentages
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter())

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    print("Starting agreement year analysis...")
    print(f"Agreement years: {list(AGREEMENT_START_YEARS)}")
    print(f"Compute thresholds: {[f'{t:,}' for t in COMPUTE_THRESHOLDS]}")
    print(f"Simulations per year: {NUM_SIMULATIONS}\n")

    results = run_analysis()
    plot_results(results)

    print("\nAnalysis complete!")
