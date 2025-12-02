"""
Proxy Project data formatting for plots.

This module provides functions to format proxy project data for frontend plots.
The actual computation logic is in backend/classes/us_project.py.

Trajectory naming conventions:
- 'global' or 'us_no_slowdown': Largest US company trajectory WITHOUT slowdown (no cap)
- 'proxy_project': Proxy project trajectory (compute capped based on PRC covert estimate)
- 'covert': PRC covert AI R&D trajectory
- 'prc_no_slowdown': PRC trajectory if there were no slowdown (they kept pace with US)
"""

from typing import Dict, Any, Optional

from backend.paramaters import ProxyProjectParameters
from backend.classes.us_project import ProxyProject


def compute_proxy_project_compute_cap(
    years: list,
    covert_compute_percentiles: Dict[str, list],
    proxy_project_params: ProxyProjectParameters
) -> Dict[str, Any]:
    """
    Compute the compute cap for the proxy project scenario over time.

    This is a convenience wrapper around ProxyProject.compute_compute_cap()
    for backward compatibility.

    Args:
        years: List of years for the time series
        covert_compute_percentiles: Dict with keys like 'p10', 'p25', 'p50', 'p75', 'p90'
            containing lists of compute values at each percentile for each year
        proxy_project_params: ProxyProjectParameters dataclass

    Returns:
        Dictionary with:
            - 'years': List of years
            - 'compute': List of compute cap values
    """
    proxy_project = ProxyProject(params=proxy_project_params)
    return proxy_project.compute_compute_cap(years, covert_compute_percentiles)


def compute_proxy_project_trajectory(
    covert_compute_data: Optional[Dict[str, Any]],
    proxy_project_params: Optional[ProxyProjectParameters] = None
) -> Optional[Dict[str, Any]]:
    """
    Compute proxy project compute trajectory (the cap for post-agreement period).

    This is a convenience wrapper for backward compatibility that returns just
    the compute cap data formatted for plots.

    Note: The full proxy project trajectory in monte carlo results combines:
    - Largest US company compute before agreement year
    - This computed cap after agreement year

    Args:
        covert_compute_data: Cached simulation data with covert compute info
        proxy_project_params: ProxyProjectParameters (uses defaults if None)

    Returns:
        Dictionary with 'years' and 'compute' lists for the compute cap,
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

    # Use new class for computation
    if proxy_project_params is None:
        proxy_project_params = ProxyProjectParameters()

    proxy_project = ProxyProject(params=proxy_project_params)
    return proxy_project.compute_compute_cap(post_years, covert_compute_percentiles)
