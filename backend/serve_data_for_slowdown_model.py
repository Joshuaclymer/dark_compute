"""
Serve data for the AI takeoff slowdown model visualization.

This module computes AI R&D speedup trajectories using the takeoff model
to predict when key capability milestones will be reached.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
from copy import deepcopy
from backend.paramaters import (
    ProxyProject,
    Parameters,
    SimulationSettings,
    CovertProjectProperties,
    CovertProjectParameters,
    SlowdownParameters,
    PCatastropheParameters,
    SoftwareProliferation
)
from backend.classes.p_catastrophe import PCatastrophe


# Default number of Monte Carlo samples for trajectory uncertainty
DEFAULT_MC_SAMPLES = 50


def compute_research_speed_adjusted_duration(
    trajectory_times: List[float],
    ai_speedup: List[float],
    start_time: float,
    end_time: float
) -> float:
    """Compute the research-speed-adjusted duration between two time points.

    This integrates the AI R&D speedup over the handoff period to get the
    effective research time (accounting for accelerated progress).

    Args:
        trajectory_times: Time points of the trajectory
        ai_speedup: AI R&D speedup values at each time point
        start_time: Start time (e.g., AC milestone time)
        end_time: End time (e.g., SAR milestone time)

    Returns:
        Research-speed-adjusted duration in years
    """
    times = np.array(trajectory_times)
    speedup = np.array(ai_speedup)

    # Find indices within the time range
    mask = (times >= start_time) & (times <= end_time)
    if not np.any(mask):
        return end_time - start_time  # Fallback to raw duration

    relevant_times = times[mask]
    relevant_speedup = speedup[mask]

    # Integrate speedup over time using trapezoidal rule
    if len(relevant_times) < 2:
        return (end_time - start_time) * relevant_speedup[0] if len(relevant_speedup) > 0 else end_time - start_time

    # Trapezoidal integration: sum of (dt * average_speedup_in_interval)
    adjusted_duration = float(np.trapz(relevant_speedup, relevant_times))

    return adjusted_duration


def compute_p_catastrophe_from_trajectories(
    trajectories: Dict[str, Any],
    p_cat_params: PCatastropheParameters
) -> Dict[str, Any]:
    """Compute P(catastrophe) values from trajectory milestones.

    Args:
        trajectories: Dictionary containing trajectory data with milestones
        p_cat_params: PCatastropheParameters with probability anchor points

    Returns:
        Dictionary containing P(catastrophe) results for global and covert trajectories
    """
    p_cat = PCatastrophe()
    results = {
        'global': None,
        'covert': None
    }

    # Process global trajectory
    milestones_global = trajectories.get('milestones_global', {})
    trajectory_times = trajectories.get('trajectory_times', [])
    global_ai_speedup = trajectories.get('global_ai_speedup', [])

    if milestones_global and trajectory_times and global_ai_speedup:
        # AC = Automated Coder, SAR = Strong AI Researcher
        # Handoff duration is the time between AC and SAR
        ac_time = milestones_global.get('AC', {}).get('time')
        sar_time = milestones_global.get('SAR-level-experiment-selection-skill', {}).get('time')

        if ac_time and sar_time:
            # Handoff duration: AC to SAR
            handoff_duration = sar_time - ac_time

            # Research-speed-adjusted handoff duration
            adjusted_handoff_duration = compute_research_speed_adjusted_duration(
                trajectory_times, global_ai_speedup, ac_time, sar_time
            )

            p_takeover = p_cat.p_AI_takeover(adjusted_handoff_duration, p_cat_params)
            p_power_grabs = p_cat.p_human_power_grabs(handoff_duration, p_cat_params)

            results['global'] = {
                'ac_time': ac_time,
                'sar_time': sar_time,
                'handoff_duration_years': handoff_duration,
                'adjusted_handoff_duration_years': adjusted_handoff_duration,
                'p_ai_takeover': p_takeover,
                'p_human_power_grabs': p_power_grabs
            }

    # Process covert trajectory
    milestones_covert = trajectories.get('milestones_covert', {})
    covert_trajectory_times = trajectories.get('covert_trajectory_times', [])
    covert_ai_speedup = trajectories.get('covert_ai_speedup', [])

    if milestones_covert and covert_trajectory_times and covert_ai_speedup:
        ac_time = milestones_covert.get('AC', {}).get('time')
        sar_time = milestones_covert.get('SAR-level-experiment-selection-skill', {}).get('time')

        if ac_time and sar_time:
            handoff_duration = sar_time - ac_time

            adjusted_handoff_duration = compute_research_speed_adjusted_duration(
                covert_trajectory_times, covert_ai_speedup, ac_time, sar_time
            )

            p_takeover = p_cat.p_AI_takeover(adjusted_handoff_duration, p_cat_params)
            p_power_grabs = p_cat.p_human_power_grabs(handoff_duration, p_cat_params)

            results['covert'] = {
                'ac_time': ac_time,
                'sar_time': sar_time,
                'handoff_duration_years': handoff_duration,
                'adjusted_handoff_duration_years': adjusted_handoff_duration,
                'p_ai_takeover': p_takeover,
                'p_human_power_grabs': p_power_grabs
            }

    return results


def get_p_catastrophe_curve_data(p_cat_params: PCatastropheParameters) -> Dict[str, Any]:
    """Get P(catastrophe) curve data for plotting.

    Args:
        p_cat_params: PCatastropheParameters with probability anchor points

    Returns:
        Dictionary containing curve data for both AI takeover and human power grabs
    """
    p_cat = PCatastrophe()

    # Generate data points for both curves
    # Use log-spaced time points from 1 week to 20 years
    durations = np.logspace(np.log10(1/52), np.log10(20), 100)  # 1 week to 20 years

    p_takeover = [p_cat.p_AI_takeover(d, p_cat_params) for d in durations]
    p_power_grabs = [p_cat.p_human_power_grabs(d, p_cat_params) for d in durations]

    return {
        'durations': durations.tolist(),
        'p_ai_takeover': p_takeover,
        'p_human_power_grabs': p_power_grabs,
        # Include the anchor points for reference
        'anchor_points': {
            'ai_takeover': {
                '1_month': p_cat_params.p_ai_takeover_1_month,
                '1_year': p_cat_params.p_ai_takeover_1_year,
                '10_years': p_cat_params.p_ai_takeover_10_years
            },
            'human_power_grabs': {
                '1_month': p_cat_params.p_human_power_grabs_1_month,
                '1_year': p_cat_params.p_human_power_grabs_1_year,
                '10_years': p_cat_params.p_human_power_grabs_10_years
            }
        }
    }


def _run_monte_carlo_trajectories(
    covert_compute_years: Optional[List[float]],
    covert_compute_values: Optional[List[float]],
    num_samples: int = DEFAULT_MC_SAMPLES,
    seed: int = 42,
    progress_callback: Optional[callable] = None,
    software_proliferation: Optional['SoftwareProliferation'] = None
) -> Dict[str, Any]:
    """Run Monte Carlo simulations of trajectory predictions with sampled parameters.

    The returned data represents full trajectories indexed by ASI achievement time:
    - The "median" line is the complete trajectory from the run where ASI is achieved at the median time
    - The p25/p75 bands show trajectories from runs where ASI is achieved at 25th/75th percentile times
    - Milestones come from these same runs, ensuring they align with the plotted lines

    Args:
        covert_compute_years: Years for the compute trajectory (or None for global)
        covert_compute_values: Compute values at each year (or None for global)
        num_samples: Number of Monte Carlo samples
        seed: Random seed for reproducibility
        progress_callback: Optional callback function(current, total, trajectory_name) for progress updates
        software_proliferation: Optional SoftwareProliferation parameters for weight/algorithm stealing.
            If provided, the covert trajectory will steal from the global (leading) trajectory.

    Returns:
        Dictionary containing:
            - trajectory_times: Common time grid for all trajectories
            - speedup_percentiles: Dict with 'p25', 'median', 'p75' arrays (full trajectories from specific runs)
            - milestones_median: Milestones from the median ASI run
            - milestones_p25: Milestones from the 25th percentile ASI run
            - milestones_p75: Milestones from the 75th percentile ASI run
            - asi_times: Dict with ASI times for p25, median, p75
            - num_successful_samples: Number of successful MC samples
    """
    import sys
    import os

    original_dir = os.getcwd()
    takeoff_model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'takeoff_model', 'ai-futures-calculator'
    )

    try:
        os.chdir(takeoff_model_dir)
        if takeoff_model_dir not in sys.path:
            sys.path.insert(0, takeoff_model_dir)
        scripts_dir = os.path.join(takeoff_model_dir, 'scripts')
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from predict_trajectory import TrajectoryPredictor
        from progress_model import load_time_series_data, Parameters as TakeoffParameters
        from batch_rollout import _sample_from_dist, _load_config, _clip_to_param_bounds

        # Load base time series data
        default_csv = os.path.join(takeoff_model_dir, 'input_data.csv')
        base_time_series = load_time_series_data(default_csv)
        present_day = TakeoffParameters().present_day

        # Load sampling configuration
        sampling_config_path = os.path.join(takeoff_model_dir, 'config', 'sampling_config.yaml')
        sampling_config = _load_config(sampling_config_path)
        param_dists = sampling_config.get('parameters', {})

        # Initialize RNG
        rng = np.random.default_rng(seed)

        # Prepare modified time series if covert compute is provided
        if covert_compute_years is not None and covert_compute_values is not None:
            covert_years_array = np.array(covert_compute_years, dtype=float)
            covert_compute_array = np.array(covert_compute_values, dtype=float)
            first_covert_year = covert_years_array[0]
        else:
            covert_years_array = None
            covert_compute_array = None
            first_covert_year = None

        # Run Monte Carlo simulations - store full trajectories and milestones
        all_runs = []  # List of dicts: {'speedup': array, 'milestones': dict, 'asi_time': float}
        common_times = None

        for sim_idx in range(num_samples):
            try:
                # Sample parameters from distributions
                sampled_params = {}
                for name, spec in param_dists.items():
                    if name == "automation_anchors":
                        continue
                    val = _sample_from_dist(spec, rng, name)
                    clip_requested = spec.get("clip_to_bounds", True)
                    if clip_requested:
                        val = _clip_to_param_bounds(name, val)
                    sampled_params[name] = val

                # Create modified time series for covert scenario
                if covert_years_array is not None:
                    modified_inference = base_time_series.inference_compute.copy()
                    modified_experiment = base_time_series.experiment_compute.copy()

                    for i, year in enumerate(base_time_series.time):
                        if year > present_day + 1 and year >= first_covert_year:
                            prc_compute = np.interp(year, covert_years_array, covert_compute_array)
                            base_total = base_time_series.inference_compute[i] + base_time_series.experiment_compute[i]
                            if base_total > 0:
                                inference_fraction = base_time_series.inference_compute[i] / base_total
                            else:
                                inference_fraction = 0.15
                            modified_inference[i] = prc_compute * inference_fraction
                            modified_experiment[i] = prc_compute * (1 - inference_fraction)

                    inference_to_use = modified_inference
                    experiment_to_use = modified_experiment
                else:
                    inference_to_use = base_time_series.inference_compute
                    experiment_to_use = base_time_series.experiment_compute

                # Run trajectory prediction
                predictor = TrajectoryPredictor(params=TakeoffParameters(**sampled_params))

                # Prepare stealing parameters if software_proliferation is provided
                # and we're running a covert trajectory (not the global trajectory)
                stealing_kwargs = {}
                if software_proliferation is not None and covert_years_array is not None:
                    # For covert trajectories with software proliferation enabled,
                    # pass the leading (global) project's compute so the predictor
                    # can compute weight stealing and algorithm stealing effects.
                    weight_stealing_times = software_proliferation.weight_stealing_times or []
                    stealing_algorithms_up_to = software_proliferation.stealing_algorithms_up_to

                    if weight_stealing_times or stealing_algorithms_up_to:
                        # Pass leading project compute (the global/unmodified compute)
                        stealing_kwargs = {
                            'inference_compute_leading_project': base_time_series.inference_compute,
                            'experiment_compute_leading_project': base_time_series.experiment_compute,
                            'L_HUMAN_leading_project': base_time_series.L_HUMAN,
                        }
                        if weight_stealing_times:
                            stealing_kwargs['years_weights_are_stolen_from_leading_project'] = weight_stealing_times
                        if stealing_algorithms_up_to:
                            stealing_kwargs['stealing_algorithms_up_to'] = stealing_algorithms_up_to

                milestones_dict = predictor.predict_from_time_series(
                    time=base_time_series.time,
                    inference_compute=inference_to_use,
                    experiment_compute=experiment_to_use,
                    L_HUMAN=base_time_series.L_HUMAN,
                    training_compute_growth_rate=base_time_series.training_compute_growth_rate,
                    **stealing_kwargs
                )

                trajectory = predictor.get_full_trajectory()
                if trajectory and 'times' in trajectory and 'ai_sw_progress_mult_ref_present_day' in trajectory:
                    times = trajectory['times']
                    speedup = trajectory['ai_sw_progress_mult_ref_present_day']

                    if common_times is None:
                        common_times = times.copy() if hasattr(times, 'copy') else np.array(times)

                    # Extract ASI time for sorting runs
                    asi_time = float('inf')
                    if 'ASI' in milestones_dict:
                        asi_time = milestones_dict['ASI'].time

                    # Serialize milestones
                    milestones_serialized = {
                        name: {
                            'time': float(info.time),
                            'progress_level': float(info.progress_level),
                            'progress_multiplier': float(info.progress_multiplier)
                        }
                        for name, info in milestones_dict.items()
                    }

                    all_runs.append({
                        'speedup': speedup.tolist() if hasattr(speedup, 'tolist') else list(speedup),
                        'milestones': milestones_serialized,
                        'asi_time': asi_time
                    })

                # Report progress after each simulation
                if progress_callback:
                    progress_callback(sim_idx + 1, num_samples)

            except Exception as e:
                print(f"Warning: Monte Carlo sample {sim_idx} failed: {e}")
                continue

        os.chdir(original_dir)

        if not all_runs:
            return None

        # Sort runs by ASI time to find percentile trajectories
        all_runs_sorted = sorted(all_runs, key=lambda x: x['asi_time'])
        n = len(all_runs_sorted)

        # Find indices for 25th, 50th (median), and 75th percentile ASI times
        idx_p25 = max(0, int(n * 0.25) - 1)
        idx_median = max(0, int(n * 0.50) - 1)
        idx_p75 = min(n - 1, int(n * 0.75))

        # Get the full trajectories from those specific runs
        run_p25 = all_runs_sorted[idx_p25]
        run_median = all_runs_sorted[idx_median]
        run_p75 = all_runs_sorted[idx_p75]

        return {
            'trajectory_times': common_times.tolist() if hasattr(common_times, 'tolist') else list(common_times),
            'speedup_percentiles': {
                'p25': run_p25['speedup'],
                'median': run_median['speedup'],
                'p75': run_p75['speedup']
            },
            'milestones_p25': run_p25['milestones'],
            'milestones_median': run_median['milestones'],
            'milestones_p75': run_p75['milestones'],
            'asi_times': {
                'p25': run_p25['asi_time'],
                'median': run_median['asi_time'],
                'p75': run_p75['asi_time']
            },
            'num_successful_samples': len(all_runs)
        }

    except Exception as e:
        print(f"Error in _run_monte_carlo_trajectories: {e}")
        import traceback
        traceback.print_exc()
        try:
            os.chdir(original_dir)
        except:
            pass
        return None


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
    def send_status(message: str):
        if status_callback:
            status_callback(message)
    from backend.serve_data_for_dark_compute_model import _load_compute_trajectory

    # Use provided parameters or defaults
    if slowdown_params is None:
        slowdown_params = SlowdownParameters()

    num_mc_samples = slowdown_params.monte_carlo_samples
    proxy_project_params = slowdown_params.proxy_project
    p_cat_params = slowdown_params.PCatastrophe_parameters
    software_proliferation = slowdown_params.software_proliferation

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
    covert_years, covert_median = _combine_prc_and_covert_compute(covert_compute_data)

    # Extract takeoff trajectories (global + covert) - legacy single-trajectory version
    send_status("Initializing takeoff model (this takes a moment)...")
    trajectories = extract_takeoff_slowdown_trajectories(params, covert_years, covert_median)

    # Build combined covert data for the frontend plot
    combined_covert_data = None
    if covert_years and covert_median:
        combined_covert_data = {
            'years': covert_years,
            'median': covert_median
        }

    # Get largest company compute trajectory
    largest_company_data = _get_largest_company_compute()

    # Compute PRC no-slowdown trajectory (extrapolate pre-agreement growth)
    prc_no_slowdown_data, prc_no_slowdown_trajectories = _compute_prc_no_slowdown_trajectory(
        covert_compute_data, covert_years, params
    )

    # Compute proxy project compute and trajectory
    proxy_project_data, proxy_project_trajectories = _compute_proxy_project_trajectory(
        covert_compute_data, params, proxy_project_params
    )

    # === Run Monte Carlo simulations for uncertainty bands ===
    print(f"Running Monte Carlo simulations ({num_mc_samples} samples) for trajectory uncertainty...")

    # Helper to create progress callback wrapper that tracks cumulative progress across all trajectories
    trajectory_offset = [0]  # Use list to allow mutation in closure

    def make_trajectory_callback(trajectory_name: str, offset: int):
        """Create a callback for a specific trajectory that reports cumulative progress."""
        def callback(current, total):
            if progress_callback:
                progress_callback(offset + current, num_mc_samples * 4, trajectory_name)
        return callback

    # 1. Global AI R&D (no slowdown) - use global compute (None for covert compute)
    global_mc = _run_monte_carlo_trajectories(
        None, None,
        num_samples=num_mc_samples,
        seed=42,
        progress_callback=make_trajectory_callback('Global AI R&D', 0)
    )

    # 2. PRC Covert AI R&D - use the combined covert compute trajectory
    #    Pass software_proliferation to enable weight/algorithm stealing from leading project
    covert_mc = None
    if covert_years and covert_median:
        covert_mc = _run_monte_carlo_trajectories(
            covert_years, covert_median,
            num_samples=num_mc_samples,
            seed=43,
            progress_callback=make_trajectory_callback('PRC Covert AI R&D', num_mc_samples),
            software_proliferation=software_proliferation
        )

    # 3. PRC No-Slowdown - use the extrapolated pre-agreement growth trajectory
    prc_no_slowdown_mc = None
    if prc_no_slowdown_data and prc_no_slowdown_data.get('years') and prc_no_slowdown_data.get('median'):
        prc_no_slowdown_mc = _run_monte_carlo_trajectories(
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
            proxy_mc = _run_monte_carlo_trajectories(
                combined_proxy_years,
                combined_proxy_compute,
                num_samples=num_mc_samples,
                seed=45,
                progress_callback=make_trajectory_callback('Proxy Project', num_mc_samples * 3)
            )
        else:
            proxy_mc = _run_monte_carlo_trajectories(
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


def _combine_prc_and_covert_compute(covert_compute_data: Optional[Dict[str, Any]]) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """Combine pre-agreement PRC compute with post-agreement covert compute.

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


def _get_largest_company_compute() -> Optional[Dict[str, Any]]:
    """Get largest company compute trajectory (inference + experiment) from AI Futures Project data."""
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


def _compute_prc_no_slowdown_trajectory(
    covert_compute_data: Optional[Dict[str, Any]],
    covert_years: Optional[List[float]],
    params: 'Parameters'
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Compute PRC no-slowdown trajectory by extrapolating pre-agreement growth.

    Returns:
        Tuple of (prc_no_slowdown_data, prc_no_slowdown_trajectories)
    """
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


def _compute_proxy_project_trajectory(
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


def extract_takeoff_slowdown_trajectories(app_params: 'Parameters', covert_compute_years: Optional[List[float]] = None, covert_compute_median: Optional[List[float]] = None) -> Dict[str, Any]:
    """Extract compute trajectories for AI takeoff slowdown analysis.

    Returns trajectory data including:
    1. AI R&D speedup time series
    2. Milestones (AC, SAR, SIAR, TED-AI, ASI) with times and speedup values

    Uses the default takeoff model parameters and compute trajectory from
    the AI Futures Project.

    Args:
        app_params: Application parameters (for agreement year only)

    Returns:
        Dictionary with trajectory arrays and milestone information
    """
    # Define time range from agreement year forward
    agreement_year = app_params.simulation_settings.start_year

    # Predict milestones using takeoff model with its default parameters
    milestones_global = None
    trajectory_global = None

    try:
        import sys
        import os
        # Save current directory
        original_dir = os.getcwd()
        # Change to takeoff model directory
        takeoff_model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'takeoff_model', 'ai-futures-calculator')
        os.chdir(takeoff_model_dir)
        sys.path.insert(0, takeoff_model_dir)

        from predict_trajectory import TrajectoryPredictor
        from progress_model import load_time_series_data

        print(f"Loading default takeoff model compute trajectory...")

        # Load the default compute trajectory from the AI Futures Project
        # Use input_data.csv - this is what the Flask API uses by default
        default_csv = os.path.join(takeoff_model_dir, 'input_data.csv')
        time_series_data = load_time_series_data(default_csv)

        print(f"Default trajectory: years {time_series_data.time[0]} to {time_series_data.time[-1]}")
        print(f"Inference compute 2030: {time_series_data.inference_compute[time_series_data.time >= 2030][0]:.2e}")
        print(f"Experiment compute 2030: {time_series_data.experiment_compute[time_series_data.time >= 2030][0]:.2e}")

        # Use TrajectoryPredictor with the default time series
        predictor_global = TrajectoryPredictor()

        milestones_global_dict = predictor_global.predict_from_time_series(
            time=time_series_data.time,
            inference_compute=time_series_data.inference_compute,
            experiment_compute=time_series_data.experiment_compute,
            L_HUMAN=time_series_data.L_HUMAN,
            training_compute_growth_rate=time_series_data.training_compute_growth_rate
        )

        # Get full trajectory for global scenario
        trajectory_global = predictor_global.get_full_trajectory()

        # Extract key milestone times with progress_multiplier
        milestones_global = {
            name: {
                'time': float(info.time),
                'progress_level': float(info.progress_level),
                'progress_multiplier': float(info.progress_multiplier)
            }
            for name, info in milestones_global_dict.items()
        }

        print(f"Successfully predicted {len(milestones_global)} milestones")

        # Restore original directory
        os.chdir(original_dir)

    except Exception as e:
        print(f"Warning: Could not predict milestones with takeoff model: {e}")
        import traceback
        traceback.print_exc()
        # Restore directory even on error
        try:
            os.chdir(original_dir)
        except:
            pass

    # Extract AI R&D speedup time series from trajectory
    global_ai_speedup = None
    trajectory_times = None

    if trajectory_global and 'times' in trajectory_global and 'ai_sw_progress_mult_ref_present_day' in trajectory_global:
        # Convert to list if needed
        times = trajectory_global['times']
        trajectory_times = times.tolist() if hasattr(times, 'tolist') else list(times)

        speedup = trajectory_global['ai_sw_progress_mult_ref_present_day']
        global_ai_speedup = speedup.tolist() if hasattr(speedup, 'tolist') else list(speedup)

    # Compute covert trajectory if covert compute data is provided
    covert_ai_speedup = None
    covert_trajectory_times = None
    milestones_covert = None

    if covert_compute_years is not None and covert_compute_median is not None and len(covert_compute_years) > 0:
        try:
            import sys
            import os
            original_dir = os.getcwd()
            takeoff_model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'takeoff_model', 'ai-futures-calculator')
            os.chdir(takeoff_model_dir)
            if takeoff_model_dir not in sys.path:
                sys.path.insert(0, takeoff_model_dir)

            from predict_trajectory import TrajectoryPredictor
            from progress_model import load_time_series_data

            print(f"Computing covert trajectory from {len(covert_compute_years)} data points...")

            # Load the base time series data (same as global trajectory)
            default_csv = os.path.join(takeoff_model_dir, 'input_data.csv')
            base_time_series = load_time_series_data(default_csv)

            # Convert covert compute to numpy arrays
            covert_years_array = np.array(covert_compute_years, dtype=float)
            covert_compute_array = np.array(covert_compute_median, dtype=float)

            print(f"Covert compute range: {covert_compute_array[0]:.2e} to {covert_compute_array[-1]:.2e} H100e")

            # Create modified time series: use global data but replace inference/experiment
            # compute with PRC covert compute for years where we have PRC data
            #
            # IMPORTANT: We only modify compute for years AFTER present_day (2025.6)
            # to ensure the model's reference point (present_day_sw_progress_rate) is
            # identical between global and PRC trajectories. This makes the AI R&D
            # speedup curves directly comparable.
            from progress_model import Parameters as TakeoffParameters
            present_day = TakeoffParameters().present_day  # Default is 2025.6

            # Create copies of the base time series arrays
            modified_inference = base_time_series.inference_compute.copy()
            modified_experiment = base_time_series.experiment_compute.copy()

            # Debug: Show present_day compute values BEFORE modification
            present_day_idx = np.argmin(np.abs(base_time_series.time - present_day))
            print(f"DEBUG: present_day={present_day}, nearest time={base_time_series.time[present_day_idx]}")
            print(f"DEBUG: Base inference at present_day: {base_time_series.inference_compute[present_day_idx]:.2e}")
            print(f"DEBUG: Base experiment at present_day: {base_time_series.experiment_compute[present_day_idx]:.2e}")

            # For each year in the base time series, if we have PRC covert data
            # AND the year is sufficiently after present_day, substitute PRC compute
            #
            # IMPORTANT: We use year > present_day + 1 because:
            # 1. present_day is 2025.6
            # 2. The model uses log-interpolation to get values at present_day
            # 3. We need year=2026 to remain unchanged so that interpolation at 2025.6 uses global values
            first_covert_year = covert_years_array[0]
            for i, year in enumerate(base_time_series.time):
                # Only modify compute sufficiently AFTER present_day to preserve the reference
                if year > present_day + 1 and year >= first_covert_year:
                    # Interpolate PRC covert compute at this year
                    prc_compute = np.interp(year, covert_years_array, covert_compute_array)

                    # Use the same inference/experiment ratio as the base time series
                    base_total = base_time_series.inference_compute[i] + base_time_series.experiment_compute[i]
                    if base_total > 0:
                        inference_fraction = base_time_series.inference_compute[i] / base_total
                    else:
                        inference_fraction = 0.15  # fallback

                    modified_inference[i] = prc_compute * inference_fraction
                    modified_experiment[i] = prc_compute * (1 - inference_fraction)

            print(f"Modified trajectory: inference at 2030 = {modified_inference[base_time_series.time >= 2030][0]:.2e} (was {base_time_series.inference_compute[base_time_series.time >= 2030][0]:.2e})")
            print(f"Modified trajectory: experiment at 2030 = {modified_experiment[base_time_series.time >= 2030][0]:.2e} (was {base_time_series.experiment_compute[base_time_series.time >= 2030][0]:.2e})")

            # Debug: Verify present_day compute is UNCHANGED
            print(f"DEBUG: Modified inference at present_day: {modified_inference[present_day_idx]:.2e}")
            print(f"DEBUG: Modified experiment at present_day: {modified_experiment[present_day_idx]:.2e}")

            # Debug: verify PRC compute is always less than global at every point
            print(f"\n=== DEBUG: Verifying PRC < Global at all points ===")
            for i, year in enumerate(base_time_series.time):
                if year >= first_covert_year:
                    global_total = base_time_series.inference_compute[i] + base_time_series.experiment_compute[i]
                    prc_total = modified_inference[i] + modified_experiment[i]
                    is_less = prc_total < global_total
                    if not is_less or year in [2026, 2030, 2035, 2040]:
                        print(f"  Year {year}: PRC={prc_total:.2e}, Global={global_total:.2e}, PRC<Global={is_less}")
            print(f"=== END DEBUG ===\n")

            # Use TrajectoryPredictor with the modified time series
            predictor_covert = TrajectoryPredictor()

            milestones_covert_dict = predictor_covert.predict_from_time_series(
                time=base_time_series.time,
                inference_compute=modified_inference,
                experiment_compute=modified_experiment,
                L_HUMAN=base_time_series.L_HUMAN,
                training_compute_growth_rate=base_time_series.training_compute_growth_rate
            )

            # Get full trajectory for covert scenario
            trajectory_covert = predictor_covert.get_full_trajectory()

            # Extract milestones
            milestones_covert = {
                name: {
                    'time': float(info.time),
                    'progress_level': float(info.progress_level),
                    'progress_multiplier': float(info.progress_multiplier)
                }
                for name, info in milestones_covert_dict.items()
            }

            # Extract AI R&D speedup time series and times
            if trajectory_covert and 'ai_sw_progress_mult_ref_present_day' in trajectory_covert:
                speedup = trajectory_covert['ai_sw_progress_mult_ref_present_day']
                covert_ai_speedup = speedup.tolist() if hasattr(speedup, 'tolist') else list(speedup)

                # Get covert trajectory times
                if 'times' in trajectory_covert:
                    times = trajectory_covert['times']
                    covert_trajectory_times = times.tolist() if hasattr(times, 'tolist') else list(times)

            print(f"Successfully computed covert trajectory with {len(milestones_covert)} milestones")

            os.chdir(original_dir)

        except Exception as e:
            print(f"Warning: Could not compute covert trajectory: {e}")
            import traceback
            traceback.print_exc()
            try:
                os.chdir(original_dir)
            except:
                pass

    return {
        'milestones_global': milestones_global,
        'trajectory_times': trajectory_times,
        'global_ai_speedup': global_ai_speedup,
        'covert_ai_speedup': covert_ai_speedup,
        'covert_trajectory_times': covert_trajectory_times,
        'milestones_covert': milestones_covert
    }


def estimate_p_tedai_before_detection(
    simulation_results: List[Tuple[Any, Any]],
    detection_threshold: float = 2.0,
    seed: int = 42
) -> Dict[str, Any]:
    """Estimate the probability that a covert PRC project achieves TED-AI prior to detection.

    For each simulation of the dark compute model (pre-computed):
    1. Collect a time series of compute operated by the PRC covert project
    2. Run trajectory prediction using the takeoff model with Monte Carlo sampled parameters
       to find when TED-AI is achieved (accounting for parameter uncertainty)
    3. Check whether TED-AI occurs before 'detection' (when LR exceeds threshold)

    Args:
        simulation_results: Pre-computed simulation results from model.simulation_results
            (list of tuples: (covert_projects dict, detectors dict))
        detection_threshold: Likelihood ratio threshold for detection (default 2.0)
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - p_tedai_before_detection: Proportion of simulations where TED-AI occurs before detection
            - tedai_times: List of TED-AI times for each simulation (None if not achieved)
            - detection_times: List of detection times for each simulation (None if not detected)
            - tedai_before_detection_count: Count of simulations where TED-AI precedes detection
            - total_simulations: Total number of simulations run
            - tedai_achieved_count: Count of simulations where TED-AI was achieved
    """
    import sys
    import os

    num_simulations = len(simulation_results)

    # Track results
    tedai_times = []
    detection_times = []
    tedai_before_detection_count = 0
    tedai_achieved_count = 0

    # Setup for takeoff model predictions
    original_dir = os.getcwd()
    takeoff_model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'takeoff_model', 'ai-futures-calculator'
    )

    try:
        os.chdir(takeoff_model_dir)
        if takeoff_model_dir not in sys.path:
            sys.path.insert(0, takeoff_model_dir)
        scripts_dir = os.path.join(takeoff_model_dir, 'scripts')
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from predict_trajectory import TrajectoryPredictor
        from progress_model import load_time_series_data, Parameters as TakeoffParameters
        from batch_rollout import _sample_from_dist, _load_config, _clip_to_param_bounds
        import model_config as cfg

        # Load the base time series data for the takeoff model
        default_csv = os.path.join(takeoff_model_dir, 'input_data.csv')
        base_time_series = load_time_series_data(default_csv)
        present_day = TakeoffParameters().present_day

        # Load sampling configuration for Monte Carlo parameter sampling
        sampling_config_path = os.path.join(takeoff_model_dir, 'config', 'sampling_config.yaml')
        sampling_config = _load_config(sampling_config_path)
        param_dists = sampling_config.get('parameters', {})

        # Initialize RNG for parameter sampling
        rng = np.random.default_rng(seed)

        for sim_idx, (covert_projects, detectors) in enumerate(simulation_results):
            covert_project = covert_projects["prc_covert_project"]
            sim_years = covert_project.years

            # === Step 1: Extract covert compute time series ===
            # Collect operational dark compute (H100e) at each year
            covert_compute_by_year = {}
            for year in sim_years:
                operational_compute = covert_project.operational_dark_compute(year)
                covert_compute_by_year[year] = operational_compute.total_h100e_tpp()

            # === Step 2: Calculate detection time ===
            detection_year = None
            for year in sim_years:
                cumulative_lr = covert_project.get_cumulative_evidence_of_covert_project(year)
                if cumulative_lr >= detection_threshold:
                    detection_year = year
                    break

            detection_times.append(detection_year)

            # === Step 3: Predict TED-AI time using takeoff model with sampled parameters ===
            tedai_time = None

            try:
                # Build compute trajectory arrays for the covert project
                covert_years_array = np.array(sorted(covert_compute_by_year.keys()), dtype=float)
                covert_compute_array = np.array([covert_compute_by_year[y] for y in covert_years_array], dtype=float)

                # Filter to valid compute values (non-zero)
                valid_mask = covert_compute_array > 0
                if np.sum(valid_mask) < 2:
                    # Not enough data points to predict
                    tedai_times.append(None)
                    continue

                # Create modified time series: use base time series but replace compute
                # with PRC covert compute for years where we have data
                modified_inference = base_time_series.inference_compute.copy()
                modified_experiment = base_time_series.experiment_compute.copy()

                first_covert_year = covert_years_array[0]
                for i, year in enumerate(base_time_series.time):
                    # Only modify compute after present_day + 1 to preserve reference point
                    if year > present_day + 1 and year >= first_covert_year:
                        # Interpolate PRC covert compute at this year
                        prc_compute = np.interp(year, covert_years_array, covert_compute_array)

                        # Use same inference/experiment ratio as base time series
                        base_total = base_time_series.inference_compute[i] + base_time_series.experiment_compute[i]
                        if base_total > 0:
                            inference_fraction = base_time_series.inference_compute[i] / base_total
                        else:
                            inference_fraction = 0.15

                        modified_inference[i] = prc_compute * inference_fraction
                        modified_experiment[i] = prc_compute * (1 - inference_fraction)

                # === Sample parameters from distributions for this simulation ===
                sampled_params = {}
                for name, spec in param_dists.items():
                    if name == "automation_anchors":
                        continue
                    val = _sample_from_dist(spec, rng, name)
                    # Clip to bounds unless explicitly disabled
                    clip_requested = spec.get("clip_to_bounds", True)
                    if clip_requested:
                        val = _clip_to_param_bounds(name, val)
                    sampled_params[name] = val

                # Run trajectory prediction with sampled parameters
                predictor = TrajectoryPredictor(params=TakeoffParameters(**sampled_params))
                milestones = predictor.predict_from_time_series(
                    time=base_time_series.time,
                    inference_compute=modified_inference,
                    experiment_compute=modified_experiment,
                    L_HUMAN=base_time_series.L_HUMAN,
                    training_compute_growth_rate=base_time_series.training_compute_growth_rate
                )

                # Extract TED-AI time
                if 'TED-AI' in milestones:
                    tedai_time = milestones['TED-AI'].time
                    tedai_achieved_count += 1

            except Exception as e:
                print(f"Warning: Could not predict TED-AI for simulation {sim_idx}: {e}")

            tedai_times.append(tedai_time)

            # === Step 4: Compare TED-AI vs detection ===
            if tedai_time is not None:
                if detection_year is None or tedai_time < detection_year:
                    tedai_before_detection_count += 1

        os.chdir(original_dir)

    except Exception as e:
        print(f"Error in estimate_p_tedai_before_detection: {e}")
        import traceback
        traceback.print_exc()
        try:
            os.chdir(original_dir)
        except:
            pass
        raise

    # Calculate probability
    p_tedai_before_detection = tedai_before_detection_count / num_simulations if num_simulations > 0 else 0.0

    return {
        'p_tedai_before_detection': p_tedai_before_detection,
        'tedai_times': tedai_times,
        'detection_times': detection_times,
        'tedai_before_detection_count': tedai_before_detection_count,
        'total_simulations': num_simulations,
        'tedai_achieved_count': tedai_achieved_count,
        'detection_threshold': detection_threshold
    }


# Alias for the streaming endpoint - identical to get_slowdown_model_data but with progress callback
def get_slowdown_model_data_with_progress(
    cached_simulation_data: Optional[Dict[str, Any]] = None,
    slowdown_params: Optional[SlowdownParameters] = None,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """Alias for get_slowdown_model_data with explicit progress callback support.

    Used by the streaming endpoint to report progress during Monte Carlo simulations.
    """
    return get_slowdown_model_data(
        cached_simulation_data=cached_simulation_data,
        slowdown_params=slowdown_params,
        progress_callback=progress_callback
    )
