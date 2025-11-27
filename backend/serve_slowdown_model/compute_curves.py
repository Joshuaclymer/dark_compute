"""
Compute curve data for the slowdown model.

This module handles combining and processing compute trajectories,
including PRC covert compute, largest company compute, and no-slowdown
counterfactual trajectories.
"""

from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.paramaters import Parameters


def combine_prc_and_covert_compute(
    covert_compute_data: Optional[Dict[str, Any]]
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """Combine pre-agreement PRC compute with post-agreement covert compute.

    Args:
        covert_compute_data: Dictionary containing:
            - prc_compute_years: Pre-agreement years
            - prc_compute_over_time: Pre-agreement compute percentiles
            - years: Post-agreement years
            - operational_dark_compute: Post-agreement compute percentiles

    Returns:
        Tuple of (combined_years, combined_median) or (None, None) if no data
    """
    if not covert_compute_data:
        return None, None

    # Get pre-agreement PRC compute (before slowdown)
    prc_years = covert_compute_data.get('prc_compute_years', [])
    prc_compute = covert_compute_data.get('prc_compute_over_time', {})
    prc_median = prc_compute.get('median', [])

    # Get post-agreement covert compute
    post_years = covert_compute_data.get('years', [])
    operational = covert_compute_data.get('operational_dark_compute', {})
    operational_median = operational.get('median', [])

    # Combine the two: pre-agreement PRC compute + post-agreement covert compute
    if prc_years and prc_median and post_years and operational_median:
        first_post_year = post_years[0] if post_years else float('inf')
        combined_years = []
        combined_median = []

        # Add pre-agreement years (before the agreement starts)
        for i, year in enumerate(prc_years):
            if year < first_post_year:
                combined_years.append(year)
                combined_median.append(prc_median[i])

        # Add post-agreement years
        combined_years.extend(post_years)
        combined_median.extend(operational_median)

        return combined_years, combined_median
    elif post_years and operational_median:
        return post_years, operational_median

    return None, None


def get_largest_company_compute() -> Optional[Dict[str, Any]]:
    """Get largest company compute trajectory (inference + experiment) from AI Futures Project data.

    Returns:
        Dictionary with 'years' and 'compute' lists, or None on error
    """
    try:
        from backend.serve_data_for_dark_compute_model import _load_compute_trajectory
        trajectory = _load_compute_trajectory()

        # Filter to years from 2026 onwards and combine inference + experiment
        lc_years = []
        lc_compute = []
        for i, t in enumerate(trajectory['time']):
            if t >= 2026:
                lc_years.append(float(t))
                total = trajectory['inference_compute'][i] + trajectory['experiment_compute'][i]
                lc_compute.append(float(total))

        return {
            'years': lc_years,
            'compute': lc_compute
        }
    except Exception as e:
        print(f"Error loading largest company compute: {e}", flush=True)
        return None


def compute_prc_no_slowdown_trajectory(
    covert_compute_data: Optional[Dict[str, Any]],
    covert_years: Optional[List[float]],
    params: 'Parameters'
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Compute PRC no-slowdown trajectory by extrapolating pre-agreement growth.

    This represents a counterfactual where PRC compute continues to grow
    at its pre-agreement rate without any slowdown from agreements.

    Args:
        covert_compute_data: Cached simulation data with covert compute info
        covert_years: Combined covert compute years (to determine end year)
        params: Main simulation parameters

    Returns:
        Tuple of (prc_no_slowdown_data, prc_no_slowdown_trajectories)
    """
    from .trajectories import extract_takeoff_slowdown_trajectories

    if not covert_compute_data:
        return None, None

    prc_years = covert_compute_data.get('prc_compute_years', [])
    prc_compute = covert_compute_data.get('prc_compute_over_time', {})
    prc_median = prc_compute.get('median', [])

    if not prc_years or not prc_median or len(prc_years) < 2:
        return None, None

    # Calculate average growth rate from the existing data
    growth_rates = []
    for i in range(1, len(prc_median)):
        if prc_median[i-1] > 0:
            growth_rates.append(prc_median[i] / prc_median[i-1])
    avg_growth_rate = sum(growth_rates) / len(growth_rates) if growth_rates else 1.0

    # Extrapolate forward to match the end of covert_years
    end_year = covert_years[-1] if covert_years else prc_years[-1]
    extended_years = list(prc_years)
    extended_median = list(prc_median)

    current_year = prc_years[-1]
    current_compute = prc_median[-1]
    while current_year < end_year:
        current_year += 1
        current_compute = current_compute * avg_growth_rate
        extended_years.append(current_year)
        extended_median.append(current_compute)

    prc_no_slowdown_data = {
        'years': extended_years,
        'median': extended_median
    }

    # Compute AI R&D speedup trajectory
    prc_no_slowdown_trajectories = extract_takeoff_slowdown_trajectories(
        params, extended_years, extended_median
    )

    return prc_no_slowdown_data, prc_no_slowdown_trajectories
