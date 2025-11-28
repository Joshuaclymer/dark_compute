"""
Optimal compute allocation between inference and experiment compute.

This module provides utilities to automatically split total compute between
inference compute (for coding automation) and experiment compute (for running
experiments) to maximize AI progress.
"""

import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution
from typing import Optional, Tuple
import logging

from progress_model import (
    compute_coding_labor,
    compute_research_effort,
    Parameters
)
import model_config as cfg

logger = logging.getLogger(__name__)


def compute_progress_rate_from_split(
    inference_fraction: float,
    total_compute: float,
    L_HUMAN: float,
    automation_fraction: float,
    research_stock: float,
    params: Parameters
) -> float:
    """
    Compute the instantaneous progress rate given a split of total compute.

    Args:
        inference_fraction: Fraction of total compute allocated to inference [0, 1]
        total_compute: Total compute budget (H100-years)
        L_HUMAN: Human labor supply
        automation_fraction: Current automation fraction
        research_stock: Current research stock
        params: Model parameters

    Returns:
        Progress rate (OOMs/year) - negative if invalid
    """
    # Split compute
    inference_compute = inference_fraction * total_compute
    experiment_compute = (1 - inference_fraction) * total_compute

    # Ensure positive values
    if inference_compute < 0 or experiment_compute < 0:
        return -1e10

    try:
        # Stage 1: Compute coding labor
        coding_labor = compute_coding_labor(
            automation_fraction=automation_fraction,
            inference_compute=inference_compute,
            L_HUMAN=L_HUMAN,
            rho=params.rho_coding_labor,
            parallel_penalty=params.parallel_penalty,
            cognitive_normalization=1.0,
            human_only=False
        )

        if not np.isfinite(coding_labor) or coding_labor <= 0:
            return -1e10

        # Stage 2: Compute serial coding labor
        serial_coding_labor = coding_labor ** params.parallel_penalty

        # Stage 3: Compute research effort (experiment capacity * aggregate taste)
        research_effort = compute_research_effort(
            experiment_compute=experiment_compute,
            serial_coding_labor=serial_coding_labor,
            alpha_experiment_capacity=params.alpha_experiment_capacity,
            rho=params.rho_experiment_capacity,
            experiment_compute_exponent=params.experiment_compute_exponent,
            aggregate_research_taste=1.0  # Simplified
        )

        if not np.isfinite(research_effort) or research_effort <= 0:
            return -1e10

        # Stage 4: Compute progress rate
        # Simplified version - uses the software progress function
        progress_rate = (research_effort / (research_stock + 1.0)) ** (1.0 / params.r_software)

        if not np.isfinite(progress_rate):
            return -1e10

        return progress_rate

    except Exception as e:
        logger.warning(f"Error computing progress rate: {e}")
        return -1e10


def optimal_compute_split(
    total_compute: float,
    L_HUMAN: float,
    automation_fraction: float,
    research_stock: float,
    params: Optional[Parameters] = None,
    method: str = 'bounded'
) -> Tuple[float, float, float]:
    """
    Find optimal split of total compute between inference and experiment.

    Args:
        total_compute: Total compute budget (H100-years)
        L_HUMAN: Human labor supply
        automation_fraction: Current automation fraction [0, 1]
        research_stock: Current research stock
        params: Model parameters (uses defaults if None)
        method: Optimization method - 'bounded' (fast) or 'global' (thorough)

    Returns:
        Tuple of (optimal_inference_compute, optimal_experiment_compute, max_progress_rate)
    """
    if params is None:
        params = Parameters()

    # Define objective (negative because we minimize)
    def objective(inference_fraction):
        return -compute_progress_rate_from_split(
            inference_fraction=inference_fraction,
            total_compute=total_compute,
            L_HUMAN=L_HUMAN,
            automation_fraction=automation_fraction,
            research_stock=research_stock,
            params=params
        )

    if method == 'global':
        # Global optimization - more thorough but slower
        result = differential_evolution(
            objective,
            bounds=[(0.0, 1.0)],
            seed=42,
            maxiter=100,
            atol=1e-6,
            tol=1e-6
        )
        optimal_fraction = result.x[0]
        max_rate = -result.fun
    else:
        # Bounded scalar minimization - fast
        result = minimize_scalar(
            objective,
            bounds=(0.0, 1.0),
            method='bounded',
            options={'xatol': 1e-6}
        )
        optimal_fraction = result.x
        max_rate = -result.fun

    # Compute optimal allocation
    optimal_inference = optimal_fraction * total_compute
    optimal_experiment = (1 - optimal_fraction) * total_compute

    return optimal_inference, optimal_experiment, max_rate


def optimal_compute_split_trajectory(
    times: np.ndarray,
    total_compute: np.ndarray,
    L_HUMAN: np.ndarray,
    automation_fractions: Optional[np.ndarray] = None,
    research_stocks: Optional[np.ndarray] = None,
    params: Optional[Parameters] = None,
    initial_inference_fraction: float = 0.15
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute optimal compute split over a trajectory.

    Args:
        times: Time points (decimal years)
        total_compute: Total compute at each time (H100-years)
        L_HUMAN: Human labor at each time
        automation_fractions: Automation fraction at each time (optional)
        research_stocks: Research stock at each time (optional)
        params: Model parameters
        initial_inference_fraction: Initial guess for inference fraction (default: 0.15)

    Returns:
        Tuple of (inference_compute, experiment_compute) arrays
    """
    if params is None:
        params = Parameters()

    n = len(times)
    inference_compute = np.zeros(n)
    experiment_compute = np.zeros(n)

    # Default automation and research stock if not provided
    if automation_fractions is None:
        automation_fractions = np.linspace(0.01, 0.5, n)  # Simple ramp
    if research_stocks is None:
        research_stocks = np.ones(n) * 1000.0  # Constant

    # Use previous optimal fraction as initial guess for next step
    current_fraction = initial_inference_fraction

    for i in range(n):
        # Use previous fraction as starting point
        result = minimize_scalar(
            lambda f: -compute_progress_rate_from_split(
                f, total_compute[i], L_HUMAN[i],
                automation_fractions[i], research_stocks[i], params
            ),
            bounds=(0.0, 1.0),
            method='bounded',
            options={'xatol': 1e-4}  # Looser tolerance for speed
        )

        current_fraction = result.x
        inference_compute[i] = current_fraction * total_compute[i]
        experiment_compute[i] = (1 - current_fraction) * total_compute[i]

    return inference_compute, experiment_compute


def compute_allocation_heuristic(
    total_compute: float,
    L_HUMAN: float,
    automation_fraction: float,
    inference_fraction_base: float = 0.15
) -> Tuple[float, float]:
    """
    Simple heuristic for compute allocation (fast, no optimization).

    Rules:
    - Base inference fraction is 15% (typical from historical data)
    - As automation increases, allocate more to experiments (less need for coding help)
    - Scale with log of total compute to reflect diminishing returns

    Args:
        total_compute: Total compute budget (H100-years)
        L_HUMAN: Human labor supply
        automation_fraction: Current automation fraction [0, 1]
        inference_fraction_base: Base inference fraction (default: 0.15)

    Returns:
        Tuple of (inference_compute, experiment_compute)
    """
    # Reduce inference fraction as automation increases
    # At 0% automation: use base
    # At 100% automation: use base * 0.5 (still need some inference for final tasks)
    automation_adjustment = 1.0 - 0.5 * automation_fraction

    # Compute per human adjustment - more compute per human → less inference needed
    compute_per_human = total_compute / max(L_HUMAN, 1.0)
    scale_adjustment = 1.0 / (1.0 + 0.1 * np.log10(max(compute_per_human, 1.0)))

    inference_fraction = inference_fraction_base * automation_adjustment * scale_adjustment
    inference_fraction = np.clip(inference_fraction, 0.05, 0.5)  # Keep in reasonable range

    inference_compute = inference_fraction * total_compute
    experiment_compute = (1 - inference_fraction) * total_compute

    return inference_compute, experiment_compute


if __name__ == "__main__":
    # Example: Find optimal split for a single point
    print("="*80)
    print("EXAMPLE: Optimal Compute Allocation")
    print("="*80)

    total = 100000.0  # 100k H100-years
    L_HUMAN_val = 100.0
    automation = 0.3
    research_stock_val = 5000.0

    print(f"\nScenario:")
    print(f"  Total compute: {total:,.0f} H100-years")
    print(f"  Human labor: {L_HUMAN_val:.0f} engineers")
    print(f"  Automation fraction: {automation:.1%}")
    print(f"  Research stock: {research_stock_val:,.0f}")

    # Optimal split
    inf_opt, exp_opt, rate_opt = optimal_compute_split(
        total_compute=total,
        L_HUMAN=L_HUMAN_val,
        automation_fraction=automation,
        research_stock=research_stock_val
    )

    print(f"\nOptimal Allocation (maximizes progress rate):")
    print(f"  Inference compute: {inf_opt:,.0f} H100-years ({inf_opt/total:.1%})")
    print(f"  Experiment compute: {exp_opt:,.0f} H100-years ({exp_opt/total:.1%})")
    print(f"  Progress rate: {rate_opt:.4f} OOMs/year")

    # Heuristic split
    inf_heur, exp_heur = compute_allocation_heuristic(
        total_compute=total,
        L_HUMAN=L_HUMAN_val,
        automation_fraction=automation
    )

    print(f"\nHeuristic Allocation (fast approximation):")
    print(f"  Inference compute: {inf_heur:,.0f} H100-years ({inf_heur/total:.1%})")
    print(f"  Experiment compute: {exp_heur:,.0f} H100-years ({exp_heur/total:.1%})")

    # Compare different splits
    print(f"\nProgress Rate vs Inference Fraction:")
    print(f"{'Inference %':<15} {'Inference':<15} {'Experiment':<15} {'Progress Rate':<15}")
    print("-" * 60)

    for inf_frac in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        rate = compute_progress_rate_from_split(
            inference_fraction=inf_frac,
            total_compute=total,
            L_HUMAN=L_HUMAN_val,
            automation_fraction=automation,
            research_stock=research_stock_val,
            params=Parameters()
        )
        inf = inf_frac * total
        exp = (1 - inf_frac) * total
        marker = " <-- Optimal" if abs(inf_frac - inf_opt/total) < 0.01 else ""
        print(f"{inf_frac:.0%}              {inf:>10,.0f}     {exp:>10,.0f}     {rate:>10.4f}{marker}")

    print("="*80)
