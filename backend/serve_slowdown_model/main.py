"""
Main endpoint for the AI takeoff slowdown model visualization.

This module consolidates all computation for the /get_slowdown_model_data endpoint,
importing from specialized component modules.
"""

from typing import Dict, Any, Optional

from backend.paramaters import (
    Parameters,
    SimulationSettings,
    CovertProjectProperties,
    CovertProjectParameters,
    SlowdownParameters,
)

from .trajectories import extract_takeoff_slowdown_trajectories
from .monte_carlo import run_monte_carlo_trajectories, DEFAULT_MC_SAMPLES
from .p_catastrophe import (
    compute_p_catastrophe_from_trajectories,
    get_p_catastrophe_curve_data,
)
from .compute_curves import (
    combine_prc_and_covert_compute,
    get_largest_company_compute,
    compute_prc_no_slowdown_trajectory,
)
from .proxy_project import compute_proxy_project_trajectory


def get_slowdown_model_data(
    cached_simulation_data: Optional[Dict[str, Any]] = None,
    slowdown_params: Optional[SlowdownParameters] = None,
    progress_callback: Optional[callable] = None,
    status_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """Compute all data needed for the slowdown model visualization page.

    This function consolidates all the computation for the /get_slowdown_model_data endpoint,
    including:
    - Global AI R&D trajectory (no slowdown baseline)
    - PRC Covert Compute trajectory
    - PRC no-slowdown trajectory (counterfactual)
    - Proxy Project compute and trajectory
    - Largest company compute trajectory
    - P(catastrophe) calculations

    Args:
        cached_simulation_data: Optional cached simulation results containing:
            - dark_compute_model: Dict with 'years', 'operational_dark_compute', etc.
            - initial_stock: Dict with 'prc_compute_years', 'prc_compute_over_time', etc.
        slowdown_params: SlowdownParameters object containing all slowdown model settings.
            If None, uses default SlowdownParameters().
        progress_callback: Optional callback function(current, total, trajectory_name) for progress updates
        status_callback: Optional callback function(status_message) for status updates

    Returns:
        Dictionary containing all data for the slowdown model plots
    """
    # Use provided parameters or defaults
    if slowdown_params is None:
        slowdown_params = SlowdownParameters()

    num_mc_samples = slowdown_params.monte_carlo_samples
    proxy_project_params = slowdown_params.proxy_project
    p_cat_params = slowdown_params.PCatastrophe_parameters

    # Use default simulation parameters
    params = Parameters(
        simulation_settings=SimulationSettings(),
        covert_project_properties=CovertProjectProperties(),
        covert_project_parameters=CovertProjectParameters()
    )

    # Extract covert compute data from cached simulation results
    covert_compute_data = None
    if cached_simulation_data and 'dark_compute_model' in cached_simulation_data:
        dark_compute_model = cached_simulation_data['dark_compute_model']
        initial_stock = cached_simulation_data.get('initial_stock', {})
        covert_compute_data = {
            'years': dark_compute_model.get('years', []),
            'operational_dark_compute': dark_compute_model.get('operational_dark_compute', {}),
            # Include pre-agreement PRC compute data
            'prc_compute_years': initial_stock.get('prc_compute_years', []),
            'prc_compute_over_time': initial_stock.get('prc_compute_over_time', {})
        }

    # Combine pre-agreement PRC compute with post-agreement covert compute
    covert_years, covert_median = combine_prc_and_covert_compute(covert_compute_data)

    # Status update: Initializing takeoff model
    if status_callback:
        status_callback("Initializing takeoff model (this takes a moment)...")

    # Extract takeoff trajectories (global + covert) - legacy single-trajectory version
    trajectories = extract_takeoff_slowdown_trajectories(params, covert_years, covert_median)

    # Build combined covert data for the frontend plot
    combined_covert_data = None
    if covert_years and covert_median:
        combined_covert_data = {
            'years': covert_years,
            'median': covert_median
        }

    # Get largest company compute trajectory
    largest_company_data = get_largest_company_compute()

    # Compute PRC no-slowdown trajectory (extrapolate pre-agreement growth)
    prc_no_slowdown_data, prc_no_slowdown_trajectories = compute_prc_no_slowdown_trajectory(
        covert_compute_data, covert_years, params
    )

    # Compute proxy project compute and trajectory
    proxy_project_data, proxy_project_trajectories = compute_proxy_project_trajectory(
        covert_compute_data, params, proxy_project_params
    )

    # === Run Monte Carlo simulations for uncertainty bands ===
    if status_callback:
        status_callback(f"Running Monte Carlo simulations ({num_mc_samples} samples)...")
    print(f"Running Monte Carlo simulations ({num_mc_samples} samples) for trajectory uncertainty...")

    # Helper to create progress callback wrapper that tracks cumulative progress across all trajectories
    def make_trajectory_callback(trajectory_name: str, offset: int):
        """Create a callback for a specific trajectory that reports cumulative progress."""
        def callback(current, total):
            if progress_callback:
                progress_callback(offset + current, num_mc_samples * 4, trajectory_name)
        return callback

    # 1. Global AI R&D (no slowdown) - use global compute (None for covert compute)
    global_mc = run_monte_carlo_trajectories(
        None, None,
        num_samples=num_mc_samples,
        seed=42,
        progress_callback=make_trajectory_callback('Global AI R&D', 0)
    )

    # 2. PRC Covert AI R&D - use the combined covert compute trajectory
    covert_mc = None
    if covert_years and covert_median:
        covert_mc = run_monte_carlo_trajectories(
            covert_years, covert_median,
            num_samples=num_mc_samples,
            seed=43,
            progress_callback=make_trajectory_callback('PRC Covert AI R&D', num_mc_samples)
        )

    # 3. PRC No-Slowdown - use the extrapolated pre-agreement growth trajectory
    prc_no_slowdown_mc = None
    if prc_no_slowdown_data and prc_no_slowdown_data.get('years') and prc_no_slowdown_data.get('median'):
        prc_no_slowdown_mc = run_monte_carlo_trajectories(
            prc_no_slowdown_data['years'],
            prc_no_slowdown_data['median'],
            num_samples=num_mc_samples,
            seed=44,
            progress_callback=make_trajectory_callback('PRC No-Slowdown', num_mc_samples * 2)
        )

    # 4. Proxy Project - use the proxy project compute trajectory
    proxy_mc = None
    if proxy_project_data and proxy_project_data.get('years') and proxy_project_data.get('compute'):
        # Combine pre-agreement PRC compute with post-agreement proxy project compute
        prc_years = covert_compute_data.get('prc_compute_years', []) if covert_compute_data else []
        prc_compute = covert_compute_data.get('prc_compute_over_time', {}) if covert_compute_data else {}
        prc_median_list = prc_compute.get('median', [])

        proxy_years = proxy_project_data['years']
        proxy_compute = proxy_project_data['compute']

        if prc_years and prc_median_list:
            first_proxy_year = proxy_years[0] if proxy_years else float('inf')
            combined_proxy_years = []
            combined_proxy_compute = []
            for i, year in enumerate(prc_years):
                if year < first_proxy_year:
                    combined_proxy_years.append(year)
                    combined_proxy_compute.append(prc_median_list[i])
            combined_proxy_years.extend(proxy_years)
            combined_proxy_compute.extend(proxy_compute)
            proxy_mc = run_monte_carlo_trajectories(
                combined_proxy_years,
                combined_proxy_compute,
                num_samples=num_mc_samples,
                seed=45,
                progress_callback=make_trajectory_callback('Proxy Project', num_mc_samples * 3)
            )
        else:
            proxy_mc = run_monte_carlo_trajectories(
                proxy_years,
                proxy_compute,
                num_samples=num_mc_samples,
                seed=45,
                progress_callback=make_trajectory_callback('Proxy Project', num_mc_samples * 3)
            )

    print("Monte Carlo simulations complete.")

    # Report completion if we have a progress callback
    if progress_callback:
        progress_callback(num_mc_samples * 4, num_mc_samples * 4, 'complete')

    # Status update: Computing P(catastrophe)
    if status_callback:
        status_callback("Computing P(catastrophe) from trajectory milestones...")

    # Compute P(catastrophe) from trajectory milestones
    p_catastrophe_results = compute_p_catastrophe_from_trajectories(trajectories, p_cat_params)

    # Get P(catastrophe) curve data for plotting
    p_catastrophe_curves = get_p_catastrophe_curve_data(p_cat_params)

    return {
        'takeoff_trajectories': trajectories,
        'agreement_year': params.simulation_settings.start_year,
        'covert_compute_data': covert_compute_data,
        'combined_covert_compute': combined_covert_data,
        'largest_company_compute': largest_company_data,
        'prc_no_slowdown_compute': prc_no_slowdown_data,
        'prc_no_slowdown_trajectories': prc_no_slowdown_trajectories,
        'proxy_project_compute': proxy_project_data,
        'proxy_project_trajectories': proxy_project_trajectories,
        # Monte Carlo uncertainty data
        'monte_carlo': {
            'global': global_mc,
            'covert': covert_mc,
            'prc_no_slowdown': prc_no_slowdown_mc,
            'proxy_project': proxy_mc
        },
        # P(catastrophe) data
        'p_catastrophe': p_catastrophe_results,
        'p_catastrophe_curves': p_catastrophe_curves
    }


def get_slowdown_model_data_with_progress(
    cached_simulation_data: Optional[Dict[str, Any]] = None,
    slowdown_params: Optional[SlowdownParameters] = None,
    progress_callback: Optional[callable] = None,
    status_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """Alias for get_slowdown_model_data with explicit progress callback support.

    Used by the streaming endpoint to report progress during Monte Carlo simulations.
    """
    return get_slowdown_model_data(
        cached_simulation_data=cached_simulation_data,
        slowdown_params=slowdown_params,
        progress_callback=progress_callback,
        status_callback=status_callback
    )
