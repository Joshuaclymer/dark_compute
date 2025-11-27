"""
Proxy project calculations for the slowdown model.

This module handles computing the compute trajectory and AI R&D speedup
for a "proxy project" that approximates PRC covert capabilities based
on a percentile of the covert compute distribution.
"""

from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING

from backend.paramaters import ProxyProject

if TYPE_CHECKING:
    from backend.paramaters import Parameters


def compute_proxy_project_compute(
    years: List[float],
    covert_compute_percentiles: Dict[str, List[float]],
    proxy_params: ProxyProject
) -> Dict[str, Any]:
    """Compute the compute available to a proxy project over time.

    The proxy project's compute cap is determined by taking a given percentile
    of the PRC operational covert compute distribution. The cap is updated
    at a specified frequency, creating a discrete/step-change curve.

    Args:
        years: List of years for the time series
        covert_compute_percentiles: Dict with keys like 'p10', 'p25', 'p50', 'p75', 'p90'
            containing lists of compute values at each percentile for each year
        proxy_params: ProxyProject dataclass containing:
            - compute_cap_as_percentile_of_PRC_operational_covert_compute: float (0-1)
            - frequency_cap_is_updated_in_years: float

    Returns:
        Dictionary with:
            - 'years': List of years
            - 'compute': List of compute values for the proxy project
    """
    if not years or not covert_compute_percentiles:
        return {'years': [], 'compute': []}

    percentile = proxy_params.compute_cap_as_percentile_of_PRC_operational_covert_compute
    update_frequency = proxy_params.frequency_cap_is_updated_in_years

    # Determine which percentile key to use based on the percentile value
    # Map the percentile (0-1) to the available keys
    percentile_keys = {
        0.10: 'p10',
        0.25: 'p25',
        0.50: 'p50',
        0.75: 'p75',
        0.90: 'p90'
    }

    # Find the closest available percentile
    available_percentiles = sorted(percentile_keys.keys())
    closest_percentile = min(available_percentiles, key=lambda x: abs(x - percentile))
    percentile_key = percentile_keys[closest_percentile]

    # Get the covert compute values at the selected percentile
    if percentile_key not in covert_compute_percentiles:
        # Fallback to median if the exact percentile isn't available
        percentile_key = 'p50' if 'p50' in covert_compute_percentiles else 'median'

    if percentile_key not in covert_compute_percentiles:
        return {'years': [], 'compute': []}

    covert_compute_at_percentile = covert_compute_percentiles[percentile_key]

    # Build the step-change curve
    # The proxy project updates its cap at the specified frequency
    proxy_compute = []
    current_cap = None
    last_update_year = None

    for i, year in enumerate(years):
        if last_update_year is None or (year - last_update_year) >= update_frequency:
            # Update the cap
            current_cap = covert_compute_at_percentile[i]
            last_update_year = year

        proxy_compute.append(current_cap if current_cap is not None else 0)

    return {
        'years': list(years),
        'compute': proxy_compute
    }


def compute_proxy_project_trajectory(
    covert_compute_data: Optional[Dict[str, Any]],
    params: 'Parameters',
    proxy_project_params: Optional[ProxyProject] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Compute proxy project compute and AI R&D speedup trajectory.

    The proxy project's cap is based on a percentile of the PRC covert compute distribution
    and is updated at a specified frequency (creating step changes).

    Args:
        covert_compute_data: Cached simulation data with covert compute info
        params: Main simulation parameters
        proxy_project_params: ProxyProject parameters (uses defaults if None)

    Returns:
        Tuple of (proxy_project_data, proxy_project_trajectories)
    """
    from .trajectories import extract_takeoff_slowdown_trajectories

    if not covert_compute_data:
        return None, None

    post_years = covert_compute_data.get('years', [])
    operational = covert_compute_data.get('operational_dark_compute', {})

    # Build percentile dict from operational dark compute data
    covert_compute_percentiles = {}
    if 'p25' in operational:
        covert_compute_percentiles['p25'] = operational['p25']
    if 'median' in operational:
        covert_compute_percentiles['p50'] = operational['median']
    if 'p75' in operational:
        covert_compute_percentiles['p75'] = operational['p75']

    if not post_years or not covert_compute_percentiles:
        return None, None

    # Compute proxy project compute (use provided params or defaults)
    if proxy_project_params is None:
        proxy_project_params = ProxyProject()
    proxy_project_data = compute_proxy_project_compute(
        post_years,
        covert_compute_percentiles,
        proxy_project_params
    )

    if not proxy_project_data or not proxy_project_data.get('years') or not proxy_project_data.get('compute'):
        return proxy_project_data, None

    # Compute proxy project AI R&D speedup trajectory
    # Need to combine pre-agreement PRC compute with post-agreement proxy project compute
    prc_years = covert_compute_data.get('prc_compute_years', [])
    prc_compute = covert_compute_data.get('prc_compute_over_time', {})
    prc_median = prc_compute.get('median', [])

    proxy_years = proxy_project_data['years']
    proxy_compute = proxy_project_data['compute']

    # Combine: pre-agreement PRC compute + post-agreement proxy project compute
    if prc_years and prc_median:
        first_proxy_year = proxy_years[0] if proxy_years else float('inf')
        combined_proxy_years = []
        combined_proxy_compute = []

        # Add pre-agreement years (before the agreement starts)
        for i, year in enumerate(prc_years):
            if year < first_proxy_year:
                combined_proxy_years.append(year)
                combined_proxy_compute.append(prc_median[i])

        # Add post-agreement proxy project years
        combined_proxy_years.extend(proxy_years)
        combined_proxy_compute.extend(proxy_compute)

        proxy_project_trajectories = extract_takeoff_slowdown_trajectories(
            params, combined_proxy_years, combined_proxy_compute
        )
    else:
        # No pre-agreement data, just use proxy project data
        proxy_project_trajectories = extract_takeoff_slowdown_trajectories(
            params, proxy_years, proxy_compute
        )

    return proxy_project_data, proxy_project_trajectories
