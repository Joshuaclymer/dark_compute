"""
Main endpoint for the AI takeoff slowdown model visualization.

This module consolidates all computation for the /get_slowdown_model_data endpoint,
importing from specialized component modules.
"""

import time
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from backend.paramaters import (
    ModelParameters,
    SimulationSettings,
    CovertProjectProperties,
    CovertProjectParameters,
    SlowdownPageParameters,
)

from .monte_carlo import (
    run_monte_carlo_trajectories,
    run_deterministic_trajectory,
    run_monte_carlo_batch_with_progress,
    DEFAULT_MC_SAMPLES,
)
from .p_catastrophe import (
    compute_p_catastrophe_from_trajectories,
    get_p_catastrophe_curve_data,
    compute_p_catastrophe_over_time,
    compute_optimal_compute_cap_over_time,
    compute_risk_reduction_over_time,
)
from .compute_curves import (
    combine_prc_and_covert_compute,
    get_largest_company_compute,
    compute_prc_no_slowdown_trajectory,
)
from .proxy_project import compute_proxy_project_trajectory


def _build_trajectories_from_mc(
    global_mc: Optional[Dict[str, Any]],
    covert_mc: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Convert Monte Carlo results to the trajectory format expected by P(catastrophe) computation.

    Args:
        global_mc: Monte Carlo results for global trajectory
        covert_mc: Monte Carlo results for covert trajectory

    Returns:
        Dictionary in the format expected by compute_p_catastrophe_from_trajectories
    """
    result = {}

    # Extract global trajectory data from Monte Carlo results
    if global_mc:
        result['trajectory_times'] = global_mc.get('trajectory_times', [])
        result['global_ai_speedup'] = global_mc.get('speedup_percentiles', {}).get('median', [])
        result['milestones_global'] = global_mc.get('milestones_median', {})

    # Extract covert trajectory data from Monte Carlo results
    if covert_mc:
        result['covert_trajectory_times'] = covert_mc.get('trajectory_times', [])
        result['covert_ai_speedup'] = covert_mc.get('speedup_percentiles', {}).get('median', [])
        result['milestones_covert'] = covert_mc.get('milestones_median', {})

    return result


def get_slowdown_model_data(
    cached_simulation_data: Optional[Dict[str, Any]] = None,
    slowdown_params: Optional[SlowdownPageParameters] = None,
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
        slowdown_params: SlowdownPageParameters object containing all slowdown model settings.
            If None, uses default SlowdownPageParameters().
        progress_callback: Optional callback function(current, total, trajectory_name) for progress updates
        status_callback: Optional callback function(status_message) for status updates

    Returns:
        Dictionary containing all data for the slowdown model plots
    """
    # Use provided parameters or defaults
    if slowdown_params is None:
        slowdown_params = SlowdownPageParameters()

    num_mc_samples = slowdown_params.monte_carlo_samples
    proxy_project_params = slowdown_params.proxy_project
    p_cat_params = slowdown_params.PCatastrophe_parameters

    # Use default simulation parameters
    params = ModelParameters(
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

    # Build combined covert data for the frontend plot
    combined_covert_data = None
    if covert_years and covert_median:
        combined_covert_data = {
            'years': covert_years,
            'median': covert_median
        }

    # Get largest company compute trajectory
    largest_company_data = get_largest_company_compute()

    # Compute PRC no-slowdown compute (extrapolate pre-agreement growth)
    prc_no_slowdown_data = compute_prc_no_slowdown_trajectory(
        covert_compute_data, covert_years
    )

    # Compute proxy project compute
    proxy_project_data = compute_proxy_project_trajectory(
        covert_compute_data, proxy_project_params
    )

    # === Run Monte Carlo simulations for uncertainty bands (in parallel) ===
    if status_callback:
        status_callback(f"Running Monte Carlo simulations ({num_mc_samples} samples) in parallel...")
    print(f"Running Monte Carlo simulations ({num_mc_samples} samples) for trajectory uncertainty (parallel)...")

    # Use time-based seed for different results each run
    base_seed = int(time.time())

    # Prepare all trajectory configs for parallel execution
    trajectory_configs = []

    # 1. Global AI R&D (no slowdown)
    trajectory_configs.append({
        'name': 'global',
        'years': None,
        'values': None,
        'seed': base_seed
    })

    # 2. PRC Covert AI R&D
    if covert_years and covert_median:
        trajectory_configs.append({
            'name': 'covert',
            'years': covert_years,
            'values': covert_median,
            'seed': base_seed + 1
        })

    # 3. PRC No-Slowdown
    if prc_no_slowdown_data and prc_no_slowdown_data.get('years') and prc_no_slowdown_data.get('median'):
        trajectory_configs.append({
            'name': 'prc_no_slowdown',
            'years': prc_no_slowdown_data['years'],
            'values': prc_no_slowdown_data['median'],
            'seed': base_seed + 2
        })

    # 4. US Frontier (proxy_project) - use largest company compute until agreement year,
    #    then proxy project compute after (matches the compute over time plot)
    if proxy_project_data and proxy_project_data.get('years') and proxy_project_data.get('compute'):
        agreement_year = params.simulation_settings.start_agreement_at_specific_year

        # Get largest company compute for pre-agreement period
        lc_years = largest_company_data.get('years', []) if largest_company_data else []
        lc_compute = largest_company_data.get('compute', []) if largest_company_data else []

        proxy_years = proxy_project_data['years']
        proxy_compute = proxy_project_data['compute']

        # Build combined trajectory: largest company until agreement, then proxy project after
        combined_proxy_years = []
        combined_proxy_compute = []

        # Add largest company compute up to and including agreement year
        for i, year in enumerate(lc_years):
            if year <= agreement_year:
                combined_proxy_years.append(year)
                combined_proxy_compute.append(lc_compute[i])

        # Add proxy project compute after agreement year
        for i, year in enumerate(proxy_years):
            if year > agreement_year:
                combined_proxy_years.append(year)
                combined_proxy_compute.append(proxy_compute[i])

        if combined_proxy_years:
            trajectory_configs.append({
                'name': 'proxy_project',
                'years': combined_proxy_years,
                'values': combined_proxy_compute,
                'seed': base_seed + 3
            })

    # Run trajectories sequentially, each using full parallelism via joblib
    # (nested parallelism with ThreadPoolExecutor + joblib causes thrashing)
    mc_results = {}

    # Map internal names to display names
    trajectory_display_names = {
        'global': 'Largest U.S. Company',
        'covert': 'PRC Covert AI R&D',
        'prc_no_slowdown': 'PRC (no slowdown)',
        'proxy_project': 'US Frontier'
    }

    total_trajectories = len(trajectory_configs)
    for idx, config in enumerate(trajectory_configs):
        display_name = trajectory_display_names.get(config['name'], config['name'])

        # Report which trajectory is starting
        if progress_callback:
            progress_callback(
                idx,
                total_trajectories,
                f"Running {display_name}... ({idx}/{total_trajectories})"
            )

        # Run MC simulation with full parallelism
        result = run_monte_carlo_trajectories(
            config['years'],
            config['values'],
            num_samples=num_mc_samples,
            seed=config['seed'],
            progress_callback=None
        )
        mc_results[config['name']] = result

        # Report completion
        if progress_callback:
            progress_callback(
                idx + 1,
                total_trajectories,
                f"Completed {display_name} ({idx + 1}/{total_trajectories})"
            )

    # Extract results
    global_mc = mc_results.get('global')
    covert_mc = mc_results.get('covert')
    prc_no_slowdown_mc = mc_results.get('prc_no_slowdown')
    proxy_mc = mc_results.get('proxy_project')

    print("Monte Carlo simulations complete.")

    # Report completion if we have a progress callback
    num_trajectories = len(trajectory_configs)
    if progress_callback:
        progress_callback(num_trajectories, num_trajectories, 'All trajectories complete')

    # Status update: Computing P(catastrophe)
    if status_callback:
        status_callback("Computing P(catastrophe) from trajectory milestones...")

    # Convert Monte Carlo data to trajectory format for P(catastrophe) computation
    # Use the median trajectory from global and covert Monte Carlo runs
    trajectories_for_p_cat = _build_trajectories_from_mc(global_mc, covert_mc)

    # Compute P(catastrophe) from trajectory milestones
    p_catastrophe_results = compute_p_catastrophe_from_trajectories(trajectories_for_p_cat, p_cat_params)

    # Get P(catastrophe) curve data for plotting
    p_catastrophe_curves = get_p_catastrophe_curve_data(p_cat_params)

    # Compute P(catastrophe) over time during the slowdown (using US Frontier trajectory)
    p_catastrophe_over_time = compute_p_catastrophe_over_time(
        proxy_mc,
        params.simulation_settings.start_agreement_at_specific_year,
        p_cat_params
    )

    # Compute optimal compute cap over time
    optimal_compute_cap_over_time = compute_optimal_compute_cap_over_time(
        p_catastrophe_over_time,
        covert_compute_data,
        params.simulation_settings.start_agreement_at_specific_year
    )

    return {
        'agreement_year': params.simulation_settings.start_agreement_at_specific_year,
        'covert_compute_data': covert_compute_data,
        'combined_covert_compute': combined_covert_data,
        'largest_company_compute': largest_company_data,
        'prc_no_slowdown_compute': prc_no_slowdown_data,
        'proxy_project_compute': proxy_project_data,
        # Monte Carlo uncertainty data
        'monte_carlo': {
            'global': global_mc,
            'covert': covert_mc,
            'prc_no_slowdown': prc_no_slowdown_mc,
            'proxy_project': proxy_mc
        },
        # P(catastrophe) data
        'p_catastrophe': p_catastrophe_results,
        'p_catastrophe_curves': p_catastrophe_curves,
        'p_catastrophe_over_time': p_catastrophe_over_time,
        'optimal_compute_cap_over_time': optimal_compute_cap_over_time
    }


def get_slowdown_model_data_with_progress(
    cached_simulation_data: Optional[Dict[str, Any]] = None,
    slowdown_params: Optional[SlowdownPageParameters] = None,
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


def _run_deterministic_wrapper(args: Tuple) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Wrapper for running deterministic trajectory in parallel."""
    name, years, values = args
    result = run_deterministic_trajectory(years, values)
    return name, result


def get_trajectory_data_fast(
    cached_simulation_data: Optional[Dict[str, Any]] = None,
    slowdown_params: Optional[SlowdownPageParameters] = None,
) -> Dict[str, Any]:
    """Get trajectory data quickly using deterministic runs (no MC uncertainty).

    This runs all 4 trajectories in parallel using joblib for maximum speed.
    Used for the AI R&D Speedup Trajectory plot which only needs medians.

    Args:
        cached_simulation_data: Optional cached simulation results
        slowdown_params: SlowdownPageParameters object

    Returns:
        Dictionary containing trajectory data with medians only (no MC bands)
    """
    if slowdown_params is None:
        slowdown_params = SlowdownPageParameters()

    proxy_project_params = slowdown_params.proxy_project
    p_cat_params = slowdown_params.PCatastrophe_parameters

    params = ModelParameters(
        simulation_settings=SimulationSettings(),
        covert_project_properties=CovertProjectProperties(),
        covert_project_parameters=CovertProjectParameters()
    )

    # Extract covert compute data
    covert_compute_data = None
    if cached_simulation_data and 'dark_compute_model' in cached_simulation_data:
        dark_compute_model = cached_simulation_data['dark_compute_model']
        initial_stock = cached_simulation_data.get('initial_stock', {})
        covert_compute_data = {
            'years': dark_compute_model.get('years', []),
            'operational_dark_compute': dark_compute_model.get('operational_dark_compute', {}),
            'prc_compute_years': initial_stock.get('prc_compute_years', []),
            'prc_compute_over_time': initial_stock.get('prc_compute_over_time', {})
        }

    covert_years, covert_median = combine_prc_and_covert_compute(covert_compute_data)
    combined_covert_data = {'years': covert_years, 'median': covert_median} if covert_years and covert_median else None

    largest_company_data = get_largest_company_compute()
    prc_no_slowdown_data = compute_prc_no_slowdown_trajectory(covert_compute_data, covert_years)
    proxy_project_data = compute_proxy_project_trajectory(covert_compute_data, proxy_project_params)

    # Build trajectory configs
    trajectory_args = []
    agreement_year = params.simulation_settings.start_agreement_at_specific_year

    # 1. Global (no slowdown)
    trajectory_args.append(('global', None, None))

    # 2. PRC Covert
    if covert_years and covert_median:
        trajectory_args.append(('covert', covert_years, covert_median))

    # 3. PRC No-Slowdown
    if prc_no_slowdown_data and prc_no_slowdown_data.get('years') and prc_no_slowdown_data.get('median'):
        trajectory_args.append(('prc_no_slowdown', prc_no_slowdown_data['years'], prc_no_slowdown_data['median']))

    # 4. US Frontier (proxy_project)
    if proxy_project_data and proxy_project_data.get('years') and proxy_project_data.get('compute'):
        lc_years = largest_company_data.get('years', []) if largest_company_data else []
        lc_compute = largest_company_data.get('compute', []) if largest_company_data else []
        proxy_years = proxy_project_data['years']
        proxy_compute = proxy_project_data['compute']

        combined_proxy_years = []
        combined_proxy_compute = []
        for i, year in enumerate(lc_years):
            if year <= agreement_year:
                combined_proxy_years.append(year)
                combined_proxy_compute.append(lc_compute[i])
        for i, year in enumerate(proxy_years):
            if year > agreement_year:
                combined_proxy_years.append(year)
                combined_proxy_compute.append(proxy_compute[i])

        if combined_proxy_years:
            trajectory_args.append(('proxy_project', combined_proxy_years, combined_proxy_compute))

    # Run all 4 trajectories in parallel using joblib
    print(f"Running {len(trajectory_args)} deterministic trajectories in parallel...", flush=True)
    num_workers = max(1, mp.cpu_count() - 1)

    if JOBLIB_AVAILABLE and len(trajectory_args) > 1:
        results = Parallel(n_jobs=min(num_workers, len(trajectory_args)), backend='loky', verbose=0)(
            delayed(_run_deterministic_wrapper)(args) for args in trajectory_args
        )
        trajectory_results = {name: result for name, result in results if result is not None}
    else:
        trajectory_results = {}
        for args in trajectory_args:
            name, result = _run_deterministic_wrapper(args)
            if result is not None:
                trajectory_results[name] = result

    print(f"Completed {len(trajectory_results)} trajectories", flush=True)

    # Convert deterministic results to MC-like format (for frontend compatibility)
    def to_mc_format(det_result):
        if not det_result:
            return None
        return {
            'trajectory_times': det_result.get('trajectory_times', []),
            'speedup_percentiles': {
                'p25': det_result.get('speedup', []),
                'median': det_result.get('speedup', []),
                'p75': det_result.get('speedup', [])
            },
            'milestones_median': det_result.get('milestones', {}),
            'asi_times': {'median': det_result.get('asi_time', 1e308)}
        }

    global_mc = to_mc_format(trajectory_results.get('global'))
    covert_mc = to_mc_format(trajectory_results.get('covert'))
    prc_no_slowdown_mc = to_mc_format(trajectory_results.get('prc_no_slowdown'))
    proxy_mc = to_mc_format(trajectory_results.get('proxy_project'))

    # Compute P(catastrophe)
    trajectories_for_p_cat = _build_trajectories_from_mc(global_mc, covert_mc)
    p_catastrophe_results = compute_p_catastrophe_from_trajectories(trajectories_for_p_cat, p_cat_params)
    p_catastrophe_curves = get_p_catastrophe_curve_data(p_cat_params)

    # Compute P(catastrophe) over time during the slowdown (using US Frontier trajectory)
    p_catastrophe_over_time = compute_p_catastrophe_over_time(
        proxy_mc,
        params.simulation_settings.start_agreement_at_specific_year,
        p_cat_params
    )

    # Compute optimal compute cap over time
    optimal_compute_cap_over_time = compute_optimal_compute_cap_over_time(
        p_catastrophe_over_time,
        covert_compute_data,
        params.simulation_settings.start_agreement_at_specific_year
    )

    # Compute risk reduction over time (slowdown vs no slowdown)
    risk_reduction_over_time = compute_risk_reduction_over_time(
        proxy_mc,  # slowdown trajectory
        global_mc,  # no-slowdown trajectory
        params.simulation_settings.start_agreement_at_specific_year,
        p_cat_params
    )

    return {
        'agreement_year': params.simulation_settings.start_agreement_at_specific_year,
        'covert_compute_data': covert_compute_data,
        'combined_covert_compute': combined_covert_data,
        'largest_company_compute': largest_company_data,
        'prc_no_slowdown_compute': prc_no_slowdown_data,
        'proxy_project_compute': proxy_project_data,
        'monte_carlo': {
            'global': global_mc,
            'covert': covert_mc,
            'prc_no_slowdown': prc_no_slowdown_mc,
            'proxy_project': proxy_mc
        },
        'p_catastrophe': p_catastrophe_results,
        'p_catastrophe_curves': p_catastrophe_curves,
        'p_catastrophe_over_time': p_catastrophe_over_time,
        'optimal_compute_cap_over_time': optimal_compute_cap_over_time,
        'risk_reduction_over_time': risk_reduction_over_time
    }


def get_uncertainty_data(
    cached_simulation_data: Optional[Dict[str, Any]] = None,
    slowdown_params: Optional[SlowdownPageParameters] = None,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """Get uncertainty data with full MC simulations for covert and proxy_project only.

    This is for the Covert Compute Prediction Uncertainty plot.
    Uses a single parallel pool for all MC samples with real-time progress updates.

    Args:
        cached_simulation_data: Optional cached simulation results
        slowdown_params: SlowdownPageParameters object
        progress_callback: Optional callback(completed, total, message) for progress updates

    Returns:
        Dictionary containing MC uncertainty data for covert and proxy_project
    """
    if slowdown_params is None:
        slowdown_params = SlowdownPageParameters()

    num_mc_samples = slowdown_params.monte_carlo_samples
    proxy_project_params = slowdown_params.proxy_project

    params = ModelParameters(
        simulation_settings=SimulationSettings(),
        covert_project_properties=CovertProjectProperties(),
        covert_project_parameters=CovertProjectParameters()
    )

    # Extract covert compute data
    covert_compute_data = None
    if cached_simulation_data and 'dark_compute_model' in cached_simulation_data:
        dark_compute_model = cached_simulation_data['dark_compute_model']
        initial_stock = cached_simulation_data.get('initial_stock', {})
        covert_compute_data = {
            'years': dark_compute_model.get('years', []),
            'operational_dark_compute': dark_compute_model.get('operational_dark_compute', {}),
            'prc_compute_years': initial_stock.get('prc_compute_years', []),
            'prc_compute_over_time': initial_stock.get('prc_compute_over_time', {})
        }

    covert_years, covert_median = combine_prc_and_covert_compute(covert_compute_data)
    largest_company_data = get_largest_company_compute()
    proxy_project_data = compute_proxy_project_trajectory(covert_compute_data, proxy_project_params)

    base_seed = int(time.time())
    agreement_year = params.simulation_settings.start_agreement_at_specific_year

    # Build trajectory configs for batch processing
    trajectory_configs = []

    if covert_years and covert_median:
        trajectory_configs.append({
            'name': 'covert',
            'years': covert_years,
            'values': covert_median,
            'seed_offset': 100  # Offset to differentiate from proxy_project seeds
        })

    if proxy_project_data and proxy_project_data.get('years') and proxy_project_data.get('compute'):
        lc_years = largest_company_data.get('years', []) if largest_company_data else []
        lc_compute = largest_company_data.get('compute', []) if largest_company_data else []
        proxy_years = proxy_project_data['years']
        proxy_compute = proxy_project_data['compute']

        combined_proxy_years = []
        combined_proxy_compute = []
        for i, year in enumerate(lc_years):
            if year <= agreement_year:
                combined_proxy_years.append(year)
                combined_proxy_compute.append(lc_compute[i])
        for i, year in enumerate(proxy_years):
            if year > agreement_year:
                combined_proxy_years.append(year)
                combined_proxy_compute.append(proxy_compute[i])

        if combined_proxy_years:
            trajectory_configs.append({
                'name': 'proxy_project',
                'years': combined_proxy_years,
                'values': combined_proxy_compute,
                'seed_offset': 200  # Different offset from covert
            })

    # Run all MC samples in a single parallel pool with real-time progress
    mc_results = run_monte_carlo_batch_with_progress(
        trajectory_configs=trajectory_configs,
        num_samples=num_mc_samples,
        base_seed=base_seed,
        progress_callback=progress_callback
    )

    return {
        'agreement_year': params.simulation_settings.start_agreement_at_specific_year,
        'monte_carlo': {
            'covert': mc_results.get('covert'),
            'proxy_project': mc_results.get('proxy_project')
        }
    }
