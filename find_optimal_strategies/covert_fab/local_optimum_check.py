"""
Local Optimum Verification Analysis

This script verifies that the proposed optimal strategy (350 operational workers,
145 construction workers, 12% SME diverted) is a good local optimum by varying
each parameter individually while holding the others fixed.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import model
from model import Model, CovertProjectStrategy

def run_local_optimum_analysis(
    base_operational_workers=728,
    base_construction_workers=448,
    base_sme_proportion=0.102,
    num_simulations=1000,
    num_test_points=10,
    results_file='local_optimum_results.pkl',
    force_rerun=False
):
    """
    Analyze the local optimum by varying each parameter around its base value.

    For each parameter, we test 10 values on a log scale within 5x of the base value
    (i.e., from base/5 to base*5), while holding the other two parameters constant.
    """

    # Check if results file exists and load if available
    if os.path.exists(results_file) and not force_rerun:
        print(f"Loading cached results from {results_file}")
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        print("Cached results loaded successfully!")
        return results

    print("Starting local optimum verification analysis...")
    print("=" * 60)
    print("Running new simulations...")
    print()

    # Initialize results storage
    results = {
        'operational_workers': {
            'values': [],
            'p_above_100k': [],
            'p_above_500k': [],
            'p_above_2m': [],
        },
        'construction_workers': {
            'values': [],
            'p_above_100k': [],
            'p_above_500k': [],
            'p_above_2m': [],
        },
        'sme_proportion': {
            'values': [],
            'p_above_100k': [],
            'p_above_500k': [],
            'p_above_2m': [],
        },
    }

    # Generate test values on log scale (5x range on each side)
    operational_values = np.logspace(
        np.log10(base_operational_workers / 5),
        np.log10(base_operational_workers * 5),
        num_test_points
    ).astype(int)
    # Add the exact base value if not already present
    operational_values = np.sort(np.unique(np.append(operational_values, base_operational_workers)))

    construction_values = np.logspace(
        np.log10(base_construction_workers / 5),
        np.log10(base_construction_workers * 5),
        num_test_points
    ).astype(int)
    # Add the exact base value if not already present
    construction_values = np.sort(np.unique(np.append(construction_values, base_construction_workers)))

    sme_values = np.logspace(
        np.log10(max(base_sme_proportion / 5, 0.001)),  # Ensure we don't go below 0.1%
        np.log10(min(base_sme_proportion * 5, 0.5)),     # Cap at 50%
        num_test_points
    )
    # Add the exact base value if not already present
    sme_values = np.sort(np.unique(np.append(sme_values, base_sme_proportion)))

    # Test 1: Vary operational workers
    print("=" * 60)
    print("Testing variation in operational workforce...")
    print("=" * 60)
    for operational_workers in operational_values:
        print(f"\nTesting operational workforce: {operational_workers} workers")
        print(f"  Construction workers: {base_construction_workers}")
        print(f"  SME proportion: {base_sme_proportion:.1%}")

        strategy = CovertProjectStrategy(
            run_a_covert_project=True,
            build_a_covert_fab=True,
            covert_fab_operating_labor=operational_workers,
            covert_fab_construction_labor=base_construction_workers,
            covert_fab_process_node="best_available_indigenously",
            covert_fab_proportion_of_prc_lithography_scanners_devoted=base_sme_proportion,
        )

        # Run simulations
        original_strategy = model.default_prc_covert_project_strategy
        model.default_prc_covert_project_strategy = strategy

        sim_model = Model(
            year_us_prc_agreement_goes_into_force=2030,
            end_year=2037,
            increment=0.1
        )
        sim_model.run_simulations(num_simulations=num_simulations)

        model.default_prc_covert_project_strategy = original_strategy

        # Calculate probabilities
        compute_at_detection = []
        for covert_projects, detectors in sim_model.simulation_results:
            us_beliefs = detectors['us_intelligence'].beliefs_about_projects['prc_covert_project']
            h100e_over_time = covert_projects['prc_covert_project'].h100e_over_time
            covert_fab = covert_projects['prc_covert_project'].covert_fab

            if covert_fab is None:
                continue

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

        if compute_at_detection:
            p_above_100k = sum(1 for c in compute_at_detection if c > 100e3) / num_simulations
            p_above_500k = sum(1 for c in compute_at_detection if c > 500e3) / num_simulations
            p_above_2m = sum(1 for c in compute_at_detection if c > 2e6) / num_simulations

            results['operational_workers']['values'].append(operational_workers)
            results['operational_workers']['p_above_100k'].append(p_above_100k)
            results['operational_workers']['p_above_500k'].append(p_above_500k)
            results['operational_workers']['p_above_2m'].append(p_above_2m)

            print(f"  P(compute > 100K): {p_above_100k:.1%}")
            print(f"  P(compute > 500K): {p_above_500k:.1%}")
            print(f"  P(compute > 2M): {p_above_2m:.1%}")

    # Test 2: Vary construction workers
    print("\n" + "=" * 60)
    print("Testing variation in construction workforce...")
    print("=" * 60)
    for construction_workers in construction_values:
        print(f"\nTesting construction workforce: {construction_workers} workers")
        print(f"  Operational workers: {base_operational_workers}")
        print(f"  SME proportion: {base_sme_proportion:.1%}")

        strategy = CovertProjectStrategy(
            run_a_covert_project=True,
            build_a_covert_fab=True,
            covert_fab_operating_labor=base_operational_workers,
            covert_fab_construction_labor=construction_workers,
            covert_fab_process_node="best_available_indigenously",
            covert_fab_proportion_of_prc_lithography_scanners_devoted=base_sme_proportion,
        )

        # Run simulations
        original_strategy = model.default_prc_covert_project_strategy
        model.default_prc_covert_project_strategy = strategy

        sim_model = Model(
            year_us_prc_agreement_goes_into_force=2030,
            end_year=2037,
            increment=0.1
        )
        sim_model.run_simulations(num_simulations=num_simulations)

        model.default_prc_covert_project_strategy = original_strategy

        # Calculate probabilities
        compute_at_detection = []
        for covert_projects, detectors in sim_model.simulation_results:
            us_beliefs = detectors['us_intelligence'].beliefs_about_projects['prc_covert_project']
            h100e_over_time = covert_projects['prc_covert_project'].h100e_over_time
            covert_fab = covert_projects['prc_covert_project'].covert_fab

            if covert_fab is None:
                continue

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

        if compute_at_detection:
            p_above_100k = sum(1 for c in compute_at_detection if c > 100e3) / num_simulations
            p_above_500k = sum(1 for c in compute_at_detection if c > 500e3) / num_simulations
            p_above_2m = sum(1 for c in compute_at_detection if c > 2e6) / num_simulations

            results['construction_workers']['values'].append(construction_workers)
            results['construction_workers']['p_above_100k'].append(p_above_100k)
            results['construction_workers']['p_above_500k'].append(p_above_500k)
            results['construction_workers']['p_above_2m'].append(p_above_2m)

            print(f"  P(compute > 100K): {p_above_100k:.1%}")
            print(f"  P(compute > 500K): {p_above_500k:.1%}")
            print(f"  P(compute > 2M): {p_above_2m:.1%}")

    # Test 3: Vary SME proportion
    print("\n" + "=" * 60)
    print("Testing variation in SME proportion diverted...")
    print("=" * 60)
    for sme_proportion in sme_values:
        print(f"\nTesting SME proportion: {sme_proportion:.1%}")
        print(f"  Operational workers: {base_operational_workers}")
        print(f"  Construction workers: {base_construction_workers}")

        strategy = CovertProjectStrategy(
            run_a_covert_project=True,
            build_a_covert_fab=True,
            covert_fab_operating_labor=base_operational_workers,
            covert_fab_construction_labor=base_construction_workers,
            covert_fab_process_node="best_available_indigenously",
            covert_fab_proportion_of_prc_lithography_scanners_devoted=sme_proportion,
        )

        # Run simulations
        original_strategy = model.default_prc_covert_project_strategy
        model.default_prc_covert_project_strategy = strategy

        sim_model = Model(
            year_us_prc_agreement_goes_into_force=2030,
            end_year=2037,
            increment=0.1
        )
        sim_model.run_simulations(num_simulations=num_simulations)

        model.default_prc_covert_project_strategy = original_strategy

        # Calculate probabilities
        compute_at_detection = []
        for covert_projects, detectors in sim_model.simulation_results:
            us_beliefs = detectors['us_intelligence'].beliefs_about_projects['prc_covert_project']
            h100e_over_time = covert_projects['prc_covert_project'].h100e_over_time
            covert_fab = covert_projects['prc_covert_project'].covert_fab

            if covert_fab is None:
                continue

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

        if compute_at_detection:
            p_above_100k = sum(1 for c in compute_at_detection if c > 100e3) / num_simulations
            p_above_500k = sum(1 for c in compute_at_detection if c > 500e3) / num_simulations
            p_above_2m = sum(1 for c in compute_at_detection if c > 2e6) / num_simulations

            results['sme_proportion']['values'].append(sme_proportion)
            results['sme_proportion']['p_above_100k'].append(p_above_100k)
            results['sme_proportion']['p_above_500k'].append(p_above_500k)
            results['sme_proportion']['p_above_2m'].append(p_above_2m)

            print(f"  P(compute > 100K): {p_above_100k:.1%}")
            print(f"  P(compute > 500K): {p_above_500k:.1%}")
            print(f"  P(compute > 2M): {p_above_2m:.1%}")

    # Save results to file
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_file}")

    return results


def plot_local_optimum_analysis(results, save_path='local_optimum_check.png'):
    """Plot the local optimum analysis results."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    ax1, ax2, ax3 = axes.flatten()

    # Plot 1: Operational workers variation
    operational_values = np.array(results['operational_workers']['values'])
    p_100k_op = np.array(results['operational_workers']['p_above_100k'])
    p_500k_op = np.array(results['operational_workers']['p_above_500k'])
    p_2m_op = np.array(results['operational_workers']['p_above_2m'])

    ax1.plot(operational_values, p_100k_op, color='darkred', linewidth=2.5,
            label='P(compute > 100K H100e)', marker='o', markersize=4)
    ax1.plot(operational_values, p_500k_op, color='#8B4513', linewidth=2.5,
            label='P(compute > 500K H100e)', marker='s', markersize=4)
    ax1.plot(operational_values, p_2m_op, color='#A0522D', linewidth=2.5,
            label='P(compute > 2M H100e)', marker='^', markersize=4)

    # Mark the base value with a vertical line
    ax1.axvline(x=728, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Optimum estimate')

    ax1.set_xlabel("Varying only 'operational workforce'", fontsize=12)
    ax1.set_ylabel('P(compute > x before detection)', fontsize=12)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=9)
    ax1.set_ylim([0, max(max(p_100k_op), max(p_500k_op), max(p_2m_op)) * 1.1])

    # Plot 2: Construction workers variation
    construction_values = np.array(results['construction_workers']['values'])
    p_100k_const = np.array(results['construction_workers']['p_above_100k'])
    p_500k_const = np.array(results['construction_workers']['p_above_500k'])
    p_2m_const = np.array(results['construction_workers']['p_above_2m'])

    ax2.plot(construction_values, p_100k_const, color='darkred', linewidth=2.5,
            label='P(compute > 100K H100e)', marker='o', markersize=4)
    ax2.plot(construction_values, p_500k_const, color='#8B4513', linewidth=2.5,
            label='P(compute > 500K H100e)', marker='s', markersize=4)
    ax2.plot(construction_values, p_2m_const, color='#A0522D', linewidth=2.5,
            label='P(compute > 2M H100e)', marker='^', markersize=4)

    # Mark the base value with a vertical line
    ax2.axvline(x=448, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Optimum estimate')

    ax2.set_xlabel("Varying only 'construction workforce'", fontsize=12)
    ax2.set_ylabel('P(compute > x before detection)', fontsize=12)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=9)
    ax2.set_ylim([0, max(max(p_100k_const), max(p_500k_const), max(p_2m_const)) * 1.1])

    # Plot 3: SME proportion variation
    sme_values = np.array(results['sme_proportion']['values']) * 100  # Convert to percentage
    p_100k_sme = np.array(results['sme_proportion']['p_above_100k'])
    p_500k_sme = np.array(results['sme_proportion']['p_above_500k'])
    p_2m_sme = np.array(results['sme_proportion']['p_above_2m'])

    ax3.plot(sme_values, p_100k_sme, color='darkred', linewidth=2.5,
            label='P(compute > 100K H100e)', marker='o', markersize=4)
    ax3.plot(sme_values, p_500k_sme, color='#8B4513', linewidth=2.5,
            label='P(compute > 500K H100e)', marker='s', markersize=4)
    ax3.plot(sme_values, p_2m_sme, color='#A0522D', linewidth=2.5,
            label='P(compute > 2M H100e)', marker='^', markersize=4)

    # Mark the base value with a vertical line
    ax3.axvline(x=10.2, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Optimum estimate')

    ax3.set_xlabel("Varying only '% domestic PRC SME diverted'", fontsize=12)
    ax3.set_ylabel('P(compute > x before detection)', fontsize=12)
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='best', fontsize=9)
    ax3.set_ylim([0, max(max(p_100k_sme), max(p_500k_sme), max(p_2m_sme)) * 1.1])

    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")

    print("\nAnalysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Run the analysis with 10 test points per parameter and 1000 simulations
    results = run_local_optimum_analysis(num_test_points=10, num_simulations=1000)

    # Generate plots
    plot_local_optimum_analysis(results)
