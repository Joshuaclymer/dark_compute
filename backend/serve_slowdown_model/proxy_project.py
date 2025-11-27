"""
Proxy project calculations for the slowdown model.

This module handles computing the compute trajectory and AI R&D speedup
for a "proxy project" that approximates PRC covert capabilities based
on a percentile of the covert compute distribution.
"""

from typing import Dict, Any, Optional, List

from backend.paramaters import ProxyProject


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
    proxy_project_params: Optional[ProxyProject] = None
) -> Optional[Dict[str, Any]]:
    """Compute proxy project compute trajectory.

    The proxy project's cap is based on a percentile of the PRC covert compute distribution
    and is updated at a specified frequency (creating step changes).

    Args:
        covert_compute_data: Cached simulation data with covert compute info
        proxy_project_params: ProxyProject parameters (uses defaults if None)

    Returns:
        Dictionary with 'years' and 'compute' lists for the proxy project compute,
        or None if insufficient data
    """
    if not covert_compute_data:
        return None

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
        return None

    # Compute proxy project compute (use provided params or defaults)
    if proxy_project_params is None:
        proxy_project_params = ProxyProject()
    proxy_project_data = compute_proxy_project_compute(
        post_years,
        covert_compute_percentiles,
        proxy_project_params
    )

    return proxy_project_data
