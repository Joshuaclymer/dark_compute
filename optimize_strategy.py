"""
Strategy Optimization Script

Uses differential evolution to find the optimal combination of:
- Operational workforce
- Construction workforce
- SME proportion

to maximize P(compute > 500K H100e before detection)
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
import model
from model import Model, CovertProjectStrategy
import pickle
import os

def evaluate_strategy(params, num_simulations=500, verbose=False):
    """
    Evaluate a strategy configuration and return negative probability
    (negative because we're minimizing, but want to maximize probability).

    Args:
        params: [operational_workers, construction_workers, sme_proportion]
        num_simulations: Number of Monte Carlo simulations
        verbose: Whether to print results

    Returns:
        Negative probability of achieving >500K H100e before detection
    """
    operational_workers = int(params[0])
    construction_workers = int(params[1])
    sme_proportion = float(params[2])

    if verbose:
        print(f"\nEvaluating: operational={operational_workers}, construction={construction_workers}, SME={sme_proportion:.3f}")

    # Create strategy
    strategy = CovertProjectStrategy(
        run_a_covert_project=True,
        build_a_covert_fab=True,
        covert_fab_operating_labor=operational_workers,
        covert_fab_construction_labor=construction_workers,
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

    # Calculate probability
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
        p_above_500k = sum(1 for c in compute_at_detection if c > 500e3) / num_simulations
    else:
        p_above_500k = 0.0

    if verbose:
        print(f"  P(compute > 500K): {p_above_500k:.3f}")

    # Return negative (since we're minimizing)
    return -p_above_500k


def optimize_with_differential_evolution(initial_guess=None, num_simulations=500):
    """
    Use differential evolution to find optimal strategy.

    This is a global optimization method that works well for noisy objectives.
    """
    print("=" * 70)
    print("OPTIMIZING STRATEGY USING DIFFERENTIAL EVOLUTION")
    print("=" * 70)
    print(f"Objective: Maximize P(compute > 500K H100e before detection)")
    print(f"Simulations per evaluation: {num_simulations}")
    print()

    if initial_guess is None:
        initial_guess = [350, 290, 0.12]

    # Define bounds for each parameter
    # Operational workers: 50-1000
    # Construction workers: 30-800
    # SME proportion: 2%-50%
    bounds = [
        (50, 1000),      # operational_workers
        (30, 800),       # construction_workers
        (0.02, 0.50)     # sme_proportion
    ]

    print("Parameter bounds:")
    print(f"  Operational workers: {bounds[0]}")
    print(f"  Construction workers: {bounds[1]}")
    print(f"  SME proportion: {bounds[2]}")
    print()
    print("Initial guess:")
    print(f"  Operational: {initial_guess[0]}, Construction: {initial_guess[1]}, SME: {initial_guess[2]:.3f}")
    print()

    # Callback to print progress
    iteration = [0]
    best_result = [float('inf')]

    def callback(xk, convergence):
        iteration[0] += 1
        current_result = evaluate_strategy(xk, num_simulations=num_simulations, verbose=False)

        if current_result < best_result[0]:
            best_result[0] = current_result
            print(f"Iteration {iteration[0]}: New best found!")
            print(f"  Operational: {int(xk[0])}, Construction: {int(xk[1])}, SME: {xk[2]:.3f}")
            print(f"  P(compute > 500K): {-current_result:.3f}")
            print()

    # Run optimization
    print("Starting optimization...")
    print()

    result = differential_evolution(
        evaluate_strategy,
        bounds,
        args=(num_simulations,),
        strategy='best1bin',
        maxiter=20,          # Maximum iterations
        popsize=5,           # Population size (smaller for faster convergence)
        tol=0.01,            # Tolerance for convergence
        mutation=(0.5, 1),   # Mutation constant
        recombination=0.7,   # Recombination constant
        seed=42,             # For reproducibility
        callback=callback,
        polish=False,        # Don't use L-BFGS-B polish (not helpful for discrete params)
        init='latinhypercube',
        atol=0.001,
        updating='deferred',
        workers=1            # Single worker (simulations are already slow)
    )

    print("=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print()
    print(f"Optimal parameters found:")
    print(f"  Operational workers: {int(result.x[0])}")
    print(f"  Construction workers: {int(result.x[1])}")
    print(f"  SME proportion: {result.x[2]:.3f} ({result.x[2]*100:.1f}%)")
    print()
    print(f"Objective value: {-result.fun:.3f}")
    print(f"  P(compute > 500K H100e): {-result.fun:.3f}")
    print()

    # Evaluate optimal solution with more simulations for final estimate
    print("Running final evaluation with 1000 simulations...")
    final_prob = -evaluate_strategy(result.x, num_simulations=1000, verbose=True)
    print()
    print(f"Final estimate with 1000 simulations: P(compute > 500K) = {final_prob:.3f}")

    return result


def optimize_with_nelder_mead(initial_guess=None, num_simulations=500):
    """
    Use Nelder-Mead simplex algorithm for optimization.

    This is a local search method that's more efficient but may get stuck in local optima.
    """
    print("=" * 70)
    print("OPTIMIZING STRATEGY USING NELDER-MEAD")
    print("=" * 70)
    print(f"Objective: Maximize P(compute > 500K H100e before detection)")
    print(f"Simulations per evaluation: {num_simulations}")
    print()

    if initial_guess is None:
        initial_guess = [350, 290, 0.12]

    print("Starting point:")
    print(f"  Operational: {initial_guess[0]}, Construction: {initial_guess[1]}, SME: {initial_guess[2]:.3f}")
    print()

    # Bounds for parameters
    bounds = [
        (50, 1000),      # operational_workers
        (30, 800),       # construction_workers
        (0.02, 0.50)     # sme_proportion
    ]

    print("Starting optimization...")
    print()

    iteration = [0]

    def callback(xk):
        iteration[0] += 1
        if iteration[0] % 5 == 0:
            result = evaluate_strategy(xk, num_simulations=num_simulations, verbose=False)
            print(f"Iteration {iteration[0]}:")
            print(f"  Operational: {int(xk[0])}, Construction: {int(xk[1])}, SME: {xk[2]:.3f}")
            print(f"  P(compute > 500K): {-result:.3f}")
            print()

    result = minimize(
        evaluate_strategy,
        initial_guess,
        args=(num_simulations,),
        method='Nelder-Mead',
        bounds=bounds,
        callback=callback,
        options={
            'maxiter': 50,
            'xatol': 1.0,      # Tolerance in parameters
            'fatol': 0.001,    # Tolerance in objective
            'adaptive': True
        }
    )

    print("=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print()
    print(f"Optimal parameters found:")
    print(f"  Operational workers: {int(result.x[0])}")
    print(f"  Construction workers: {int(result.x[1])}")
    print(f"  SME proportion: {result.x[2]:.3f} ({result.x[2]*100:.1f}%)")
    print()
    print(f"Objective value: {-result.fun:.3f}")
    print(f"  P(compute > 500K H100e): {-result.fun:.3f}")
    print()

    return result


if __name__ == "__main__":
    # Starting from our current best guess
    initial_guess = [350, 290, 0.12]

    # Use differential evolution (better for noisy objectives and global search)
    print("Using Differential Evolution optimizer...")
    print("This is well-suited for noisy objectives from Monte Carlo simulations")
    print()

    result = optimize_with_differential_evolution(
        initial_guess=initial_guess,
        num_simulations=300  # Use 300 per evaluation for speed, then verify with 1500
    )

    # Re-evaluate with 1500 simulations for final validation
    print("\n" + "=" * 70)
    print("FINAL VALIDATION WITH 1500 SIMULATIONS")
    print("=" * 70)
    final_prob = -evaluate_strategy(result.x, num_simulations=1500, verbose=True)
    print(f"\nFinal validated estimate: P(compute > 500K) = {final_prob:.4f}")

    # Save results
    results = {
        'optimal_params': result.x,
        'optimal_value': -result.fun,
        'final_validated_value': final_prob,
        'initial_guess': initial_guess,
    }

    with open('optimization_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("\nResults saved to optimization_results.pkl")
