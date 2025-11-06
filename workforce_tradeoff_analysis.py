"""
Analysis of workforce size vs expected compute produced before detection.

This script tests different PRC covert project strategies by varying the operational
workforce size while ensuring that:
1. Scanner availability is not a bottleneck (enough scanners devoted to the fab)
2. Construction labor is not a bottleneck (enough workers to complete construction on time)

We then measure the expected compute produced before detection for each strategy.
"""

import numpy as np
import matplotlib.pyplot as plt
from model import Model, CovertProjectStrategy
from fab_model import FabModelParameters
import model  # Import module to access global variables
import pickle
import os

def calculate_required_scanner_proportion(operation_labor):
    """
    Calculate the minimum proportion of scanners needed so scanners are not a bottleneck.

    From fab_model.py:
    - Labor capacity: wafers_per_month_per_worker * operation_labor
    - Scanner capacity: wafers_per_month_per_lithography_scanner * num_scanners

    We need scanner capacity >= labor capacity for scanners not to be bottleneck.

    Note: This uses expected values. Actual scanner availability varies by:
    - Process node selected (different localization years)
    - Years since localization (affects total scanners produced)
    - Random variation in scanner production (30% relative uncertainty)
    """
    # Labor capacity (with expected productivity multiplier of 1.0)
    labor_capacity = FabModelParameters.wafers_per_month_per_worker * operation_labor

    # Expected scanner count for 130nm node (most common in simulations)
    # At construction start (2028), roughly 2 years after typical localization (~2026)
    # Total scanners = 20 + 16*2 = 52 (base case, no uncertainty)
    # At operational start (~2030), roughly 4 years after localization
    # Total scanners = 20 + 16*4 = 84
    # Use operational start estimate since that's when scanners matter
    expected_total_scanners = 84

    # Required proportion: labor_capacity / (wafers_per_scanner * total_scanners)
    required_proportion = labor_capacity / (FabModelParameters.wafers_per_month_per_lithography_scanner * expected_total_scanners)

    # Add 20% buffer to account for uncertainty and ensure scanners aren't bottleneck
    # Cap at 50% - cannot divert more than half of all PRC scanners
    return min(required_proportion * 1.2, 0.5)

def calculate_required_construction_labor(operation_labor):
    """
    Calculate the optimal construction labor for minimizing detection probability.

    From fab_model.py:
    - Construction time is determined by the more constraining of:
      1. Fab capacity (log relationship)
      2. Construction labor (linear relationship)

    Empirically, we found that the optimal construction workforce is ~2.4x the minimum
    required to avoid being a bottleneck. This is because faster construction reduces
    the window for detection during the construction phase.
    """
    # Expected wafer capacity based on operation labor
    expected_wafer_capacity = FabModelParameters.wafers_per_month_per_worker * operation_labor

    # Required construction workers (from fab_model.py parameter)
    required_workers = (FabModelParameters.construction_workers_per_1000_wafers_per_month / 1000) * expected_wafer_capacity

    # Use 2.4x multiplier based on empirical optimization (local optimum check showed ~2x is optimal)
    return required_workers * 2.4

def run_workforce_analysis(num_simulations=1000, results_file='workforce_analysis_results.pkl', force_rerun=False):
    """
    Run analysis across different workforce sizes.

    Args:
        num_simulations: Number of Monte Carlo simulations per workforce size
        results_file: Path to save/load results
        force_rerun: If True, ignore cached results and rerun simulations

    Returns:
        results: Dict with workforce sizes and corresponding probabilities
    """
    # Check if results file exists and load if available
    if os.path.exists(results_file) and not force_rerun:
        print(f"Loading cached results from {results_file}")
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        print("Cached results loaded successfully!")
        return results

    print("Running new simulations...")

    # Test different operational workforce sizes (log scale from 50 to 5000, 20 values)
    workforce_sizes = np.logspace(np.log10(50), np.log10(5000), 20).astype(int)
    workforce_sizes = sorted(list(set(workforce_sizes)))  # Remove duplicates and sort

    results = {
        'workforce_sizes': workforce_sizes,
        'p_above_100k': [],
        'p_above_500k': [],
        'p_above_1m': [],
        'detection_rate': [],
        'scanner_proportions': [],
        'construction_labor': [],
        'median_construction_time': [],
        'median_production_capacity': [],
        'median_lr_inventory': []
    }

    for workforce_size in workforce_sizes:
        print(f"\nTesting workforce size: {workforce_size} workers")

        # Calculate required parameters
        scanner_proportion = calculate_required_scanner_proportion(workforce_size)
        construction_labor = calculate_required_construction_labor(workforce_size)

        # Store scanner proportion and construction labor
        results['scanner_proportions'].append(scanner_proportion)
        results['construction_labor'].append(construction_labor)

        print(f"  Scanner proportion: {scanner_proportion:.3f}")
        print(f"  Construction labor: {construction_labor:.0f} workers")

        # Temporarily update the global strategy
        original_strategy = model.default_prc_covert_project_strategy

        model.default_prc_covert_project_strategy = CovertProjectStrategy(
            run_a_covert_project=True,
            build_a_covert_fab=True,
            covert_fab_operating_labor=workforce_size,
            covert_fab_construction_labor=int(construction_labor),
            covert_fab_process_node="best_available_indigenously",  # Auto-select best available node
            covert_fab_proportion_of_prc_lithography_scanners_devoted=scanner_proportion
        )

        # Run simulations
        sim_model = Model(
            year_us_prc_agreement_goes_into_force=2030,
            end_year=2037,
            increment=0.1
        )

        sim_model.run_simulations(num_simulations=num_simulations)

        # Restore original strategy
        model.default_prc_covert_project_strategy = original_strategy

        # Extract compute at detection, construction times, production capacity, and lr_inventory
        compute_at_detection = []
        construction_times = []
        production_capacities = []
        lr_inventory_values = []

        for covert_projects, detectors in sim_model.simulation_results:
            us_beliefs = detectors['us_intelligence'].beliefs_about_projects['prc_covert_project']
            h100e_over_time = covert_projects['prc_covert_project'].h100e_over_time
            covert_fab = covert_projects['prc_covert_project'].covert_fab

            if covert_fab is None:
                continue

            # Record construction duration
            construction_times.append(covert_fab.construction_duration)

            # Record production capacity (regardless of operational status)
            production_capacities.append(covert_fab.production_capacity)

            # Find detection year
            years = sorted(us_beliefs.keys())
            detection_year = None
            for year in years:
                if us_beliefs[year].p_covert_fab_exists >= 0.5:
                    detection_year = year
                    break

            if detection_year is not None:
                h100e_at_detection = h100e_over_time.get(detection_year, 0.0)
                compute_at_detection.append(h100e_at_detection)

                # Record lr_inventory at detection
                # Need to call detection_likelihood_ratio to populate the lr_inventory attribute
                _ = covert_fab.detection_likelihood_ratio(detection_year)
                lr_inventory_values.append(covert_fab.lr_inventory)

        # Calculate probabilities of exceeding thresholds
        if compute_at_detection:
            p_above_100k = sum(1 for c in compute_at_detection if c > 100e3) / num_simulations
            p_above_500k = sum(1 for c in compute_at_detection if c > 500e3) / num_simulations
            p_above_1m = sum(1 for c in compute_at_detection if c > 2e6) / num_simulations

            results['p_above_100k'].append(p_above_100k)
            results['p_above_500k'].append(p_above_500k)
            results['p_above_1m'].append(p_above_1m)
            results['detection_rate'].append(len(compute_at_detection) / num_simulations)

            # Store median construction time
            if construction_times:
                median_construction_time = np.median(construction_times)
                results['median_construction_time'].append(median_construction_time)
            else:
                results['median_construction_time'].append(0)

            # Store median production capacity
            if production_capacities:
                median_production_capacity = np.median(production_capacities)
                results['median_production_capacity'].append(median_production_capacity)
            else:
                results['median_production_capacity'].append(0)

            # Store median lr_inventory
            if lr_inventory_values:
                median_lr_inventory = np.median(lr_inventory_values)
                results['median_lr_inventory'].append(median_lr_inventory)
            else:
                results['median_lr_inventory'].append(0)

            print(f"  P(compute > 100K): {p_above_100k:.1%}")
            print(f"  P(compute > 500K): {p_above_500k:.1%}")
            print(f"  P(compute > 1M): {p_above_1m:.1%}")
            print(f"  Detection rate: {len(compute_at_detection)/num_simulations:.1%}")
            if construction_times:
                print(f"  Median construction time: {median_construction_time:.2f} years")
            if production_capacities:
                print(f"  Median production capacity: {median_production_capacity:.0f} H100e/month")
            if lr_inventory_values:
                print(f"  Median LR inventory: {median_lr_inventory:.2f}")
        else:
            results['p_above_100k'].append(0)
            results['p_above_500k'].append(0)
            results['p_above_1m'].append(0)
            results['detection_rate'].append(0)
            results['median_construction_time'].append(0)
            results['median_production_capacity'].append(0)
            results['median_lr_inventory'].append(0)
            # Scanner proportion already stored above
            print(f"  No detections")

    # Save results to file
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_file}")

    return results

def plot_workforce_tradeoff(results, save_path='workforce_tradeoff.png'):
    """
    Create visualization of workforce size vs various metrics in a 2x3 grid.

    Args:
        results: Results dictionary from run_workforce_analysis
        save_path: Path to save the plot
    """
    workforce_sizes = results['workforce_sizes']
    p_above_100k = np.array(results['p_above_100k'])
    p_above_500k = np.array(results['p_above_500k'])
    p_above_1m = np.array(results['p_above_1m'])
    scanner_proportions = np.array(results['scanner_proportions'])
    construction_labor = np.array(results['construction_labor'])
    median_construction_time = np.array(results['median_construction_time'])
    median_production_capacity = np.array(results['median_production_capacity'])
    median_lr_inventory = np.array(results['median_lr_inventory'])

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    ax1, ax2, ax3 = axes.flatten()

    # Left subplot: Probability curves
    ax1.plot(workforce_sizes, p_above_100k, color='darkred', linewidth=2.5,
            label='P(compute > 100K H100e)', marker='o', markersize=4)
    ax1.plot(workforce_sizes, p_above_500k, color='#8B4513', linewidth=2.5,
            label='P(compute > 500K H100e)', marker='s', markersize=4)
    ax1.plot(workforce_sizes, p_above_1m, color='#A0522D', linewidth=2.5,
            label='P(compute > 2M H100e)', marker='^', markersize=4)

    # Find and label the maximum point for P(compute > 500K H100e)
    max_idx = np.argmax(p_above_500k)
    max_workforce = workforce_sizes[max_idx]
    max_prob = p_above_500k[max_idx]
    ax1.plot(max_workforce, max_prob, 'o', color='black', markersize=8, zorder=5)
    ax1.annotate(f'{max_workforce} workers\n{max_prob:.1%}',
                xy=(max_workforce, max_prob),
                xytext=(-80, 10), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1, alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax1.set_xlabel('Covert fab operational workforce size', fontsize=12)
    ax1.set_ylabel('P(compute > x before detection)', fontsize=12)
    ax1.set_xscale('log')  # Log scale for x-axis
    ax1.set_xlim([50, 1000])  # Set x-axis limit to 1000
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=10)
    ax1.set_ylim([0, 1])

    # Middle subplot: Scanner proportion diverted
    ax2.plot(workforce_sizes, scanner_proportions * 100, color='darkblue', linewidth=2.5,
            marker='o', markersize=4)

    # Label the corresponding point (same workforce size as max probability)
    scanner_prop_at_max = scanner_proportions[max_idx] * 100
    ax2.plot(max_workforce, scanner_prop_at_max, 'o', color='black', markersize=8, zorder=5)
    ax2.annotate(f'{max_workforce} workers\n{scanner_prop_at_max:.1f}% SME',
                xy=(max_workforce, scanner_prop_at_max),
                xytext=(-80, 10), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1, alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax2.set_xlabel('Covert fab operational workforce size', fontsize=12)
    ax2.set_ylabel('% domestic PRC SME diverted', fontsize=12)
    ax2.set_xscale('log')  # Log scale for x-axis
    ax2.set_xlim([50, 1000])  # Set x-axis limit to 1000
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([0, max(scanner_proportions * 100) * 1.1])  # Add 10% margin

    # Third subplot: Construction labor
    ax3.plot(workforce_sizes, construction_labor, color='darkgreen', linewidth=2.5,
            marker='o', markersize=4)

    # Label the corresponding point (same workforce size as max probability)
    construction_labor_at_max = construction_labor[max_idx]
    ax3.plot(max_workforce, construction_labor_at_max, 'o', color='black', markersize=8, zorder=5)
    ax3.annotate(f'{max_workforce} workers\n{construction_labor_at_max:.0f} workers',
                xy=(max_workforce, construction_labor_at_max),
                xytext=(-80, 10), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1, alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax3.set_xlabel('Covert fab operational workforce size', fontsize=12)
    ax3.set_ylabel('Construction workforce size', fontsize=12)
    ax3.set_xscale('log')  # Log scale for x-axis
    ax3.set_xlim([50, 1000])  # Set x-axis limit to 1000
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_ylim([0, max(construction_labor) * 1.1])  # Add 10% margin

    # # Fourth subplot: Median construction time (hidden)
    # ax4.plot(workforce_sizes, median_construction_time, color='purple', linewidth=2.5,
    #         marker='o', markersize=4)

    # ax4.set_xlabel('Covert fab operational workforce size', fontsize=12, fontweight='bold')
    # ax4.set_ylabel('Median construction time (years)', fontsize=12, fontweight='bold')
    # ax4.set_xscale('log')  # Log scale for x-axis
    # ax4.grid(True, alpha=0.3, linestyle='--')
    # if max(median_construction_time) > 0:
    #     ax4.set_ylim([0, max(median_construction_time) * 1.1])  # Add 10% margin

    # # Fifth subplot: Median production capacity (hidden)
    # ax5.plot(workforce_sizes, median_production_capacity, color='orange', linewidth=2.5,
    #         marker='o', markersize=4)

    # ax5.set_xlabel('Covert fab operational workforce size', fontsize=12, fontweight='bold')
    # ax5.set_ylabel('Median production capacity (H100e/month)', fontsize=12, fontweight='bold')
    # ax5.set_xscale('log')  # Log scale for x-axis
    # ax5.grid(True, alpha=0.3, linestyle='--')
    # if max(median_production_capacity) > 0:
    #     ax5.set_ylim([0, max(median_production_capacity) * 1.1])  # Add 10% margin

    # # Sixth subplot: Median LR inventory (hidden)
    # ax6.plot(workforce_sizes, median_lr_inventory, color='brown', linewidth=2.5,
    #         marker='o', markersize=4)

    # ax6.set_xlabel('Covert fab operational workforce size', fontsize=12, fontweight='bold')
    # ax6.set_ylabel('Median likelihood ratio (inventory)', fontsize=12, fontweight='bold')
    # ax6.set_xscale('log')  # Log scale for x-axis
    # ax6.grid(True, alpha=0.3, linestyle='--')
    # if max(median_lr_inventory) > 0:
    #     ax6.set_ylim([0, max(median_lr_inventory) * 1.1])  # Add 10% margin

    # # Add text box with key assumptions (removed)
    # textstr = 'Assumptions:\n• Scanner allocation: non-bottleneck\n• Construction labor: non-bottleneck\n• Detection threshold: 50% US probability'
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    # ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
    #         verticalalignment='top', horizontalalignment='left', bbox=props)

    plt.tight_layout(pad=1.5)  # Add extra padding to prevent label cutoff
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    print("Starting workforce size trade-off analysis...")
    print("=" * 60)

    # Run analysis
    results = run_workforce_analysis(num_simulations=1000)

    # Create visualization
    plot_workforce_tradeoff(results)

    print("\nAnalysis complete!")
    print("=" * 60)
