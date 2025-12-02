"""
Monte Carlo simulations for trajectory uncertainty.

This module runs Monte Carlo simulations of trajectory predictions
with sampled parameters to quantify uncertainty in AI R&D speedup curves.
Supports parallel execution across multiple CPU cores using joblib.
"""

import numpy as np
import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, Optional, List, Tuple
from scipy.signal import savgol_filter

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


def _smooth_speedup_trajectory(speedup: np.ndarray, window_length: int = 11, polyorder: int = 2) -> np.ndarray:
    """Apply Savitzky-Golay filter to smooth speedup trajectory.

    This removes bumps from discrete capability cap enforcement while preserving
    the overall shape and monotonicity of the trajectory.

    Args:
        speedup: Array of speedup values
        window_length: Window length for filter (must be odd, default 11)
        polyorder: Polynomial order for filter (default 2)

    Returns:
        Smoothed speedup array
    """
    if len(speedup) < window_length:
        return speedup

    # Apply Savitzky-Golay filter
    smoothed = savgol_filter(speedup, window_length, polyorder)

    # Ensure smoothed values don't go below 1 (minimum speedup)
    smoothed = np.maximum(smoothed, 1.0)

    return smoothed

# Default number of Monte Carlo samples for trajectory uncertainty
DEFAULT_MC_SAMPLES = 50

# Number of workers for parallel execution (use ALL available cores)
DEFAULT_NUM_WORKERS = mp.cpu_count()


def _get_takeoff_model_dir():
    """Get the path to the takeoff model directory."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'takeoff_model', 'ai-futures-calculator'
    )


def _setup_takeoff_model():
    """Set up the takeoff model environment and return necessary imports and data.

    Returns:
        Tuple of (takeoff_model_dir, base_time_series, present_day, TrajectoryPredictor, TakeoffParameters)
    """
    takeoff_model_dir = _get_takeoff_model_dir()

    if takeoff_model_dir not in sys.path:
        sys.path.insert(0, takeoff_model_dir)
    scripts_dir = os.path.join(takeoff_model_dir, 'scripts')
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from predict_trajectory import TrajectoryPredictor
    from progress_model import load_time_series_data, Parameters as TakeoffParameters

    # Load base time series data
    default_csv = os.path.join(takeoff_model_dir, 'input_data.csv')
    base_time_series = load_time_series_data(default_csv)
    present_day = TakeoffParameters().present_day

    return takeoff_model_dir, base_time_series, present_day, TrajectoryPredictor, TakeoffParameters


def _prepare_compute_arrays(covert_compute_years, covert_compute_values, base_time_series, present_day):
    """Prepare modified compute arrays for covert scenario.

    Returns:
        Tuple of (inference_to_use, experiment_to_use)
    """
    if covert_compute_years is not None and covert_compute_values is not None:
        covert_years_array = np.array(covert_compute_years, dtype=float)
        covert_compute_array = np.array(covert_compute_values, dtype=float)
        first_covert_year = covert_years_array[0]

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

        return modified_inference, modified_experiment
    else:
        return base_time_series.inference_compute, base_time_series.experiment_compute


def _serialize_milestones(milestones_dict):
    """Serialize milestones dict to JSON-compatible format."""
    return {
        name: {
            'time': float(info.time),
            'progress_level': float(info.progress_level),
            'progress_multiplier': float(info.progress_multiplier)
        }
        for name, info in milestones_dict.items()
    }


def _run_single_mc_sample(args):
    """Run a single Monte Carlo sample. Designed to be called in a worker process.

    Args:
        args: Tuple of (sim_idx, seed, covert_compute_years, covert_compute_values)

    Returns:
        Dictionary with speedup, milestones, asi_time, or None if failed
    """
    sim_idx, seed, covert_compute_years, covert_compute_values = args

    original_dir = os.getcwd()

    try:
        takeoff_model_dir = _get_takeoff_model_dir()
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

        # Initialize RNG with unique seed for this sample
        rng = np.random.default_rng(seed)

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

        # Prepare compute arrays
        if covert_compute_years is not None and covert_compute_values is not None:
            covert_years_array = np.array(covert_compute_years, dtype=float)
            covert_compute_array = np.array(covert_compute_values, dtype=float)
            first_covert_year = covert_years_array[0]

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

        # Run trajectory prediction with sampled parameters
        predictor = TrajectoryPredictor(params=TakeoffParameters(**sampled_params))
        milestones_dict = predictor.predict_from_time_series(
            time=base_time_series.time,
            inference_compute=inference_to_use,
            experiment_compute=experiment_to_use,
            L_HUMAN=base_time_series.L_HUMAN,
            training_compute_growth_rate=base_time_series.training_compute_growth_rate
        )

        trajectory = predictor.get_full_trajectory()
        os.chdir(original_dir)

        if trajectory and 'times' in trajectory and 'ai_sw_progress_mult_ref_present_day' in trajectory:
            speedup = trajectory['ai_sw_progress_mult_ref_present_day']

            asi_time = 1e308
            if 'ASI' in milestones_dict:
                asi_time = milestones_dict['ASI'].time

            return {
                'sim_idx': sim_idx,
                'speedup': speedup.tolist() if hasattr(speedup, 'tolist') else list(speedup),
                'milestones': _serialize_milestones(milestones_dict),
                'asi_time': asi_time
            }

        return None

    except Exception as e:
        try:
            os.chdir(original_dir)
        except:
            pass
        return None


def run_deterministic_trajectory(
    covert_compute_years: Optional[List[float]],
    covert_compute_values: Optional[List[float]],
    capability_cap: Optional[List[float]] = None,
    capability_cap_times: Optional[List[float]] = None,
    apply_cap_margin: bool = False,
) -> Optional[Dict[str, Any]]:
    """Run a single trajectory prediction with default (median) parameters.

    This is more efficient than running Monte Carlo when you just need the median trajectory.

    Args:
        covert_compute_years: Years for the compute trajectory (or None for global)
        covert_compute_values: Compute values at each year (or None for global)
        capability_cap: Optional list of progress cap values.
            If provided, progress will be upper-bounded by these values.
        capability_cap_times: Times corresponding to capability_cap values.
            If None, assumes capability_cap already matches base_time_series.time.
        apply_cap_margin: If True, applies US_SLOWDOWN_CAP_MARGIN to the capability cap
            to create visual separation between the trajectory and cap line.

    Returns:
        Dictionary containing:
            - trajectory_times: Time grid for the trajectory
            - speedup: The AI R&D speedup values
            - milestones: Milestone information
            - asi_time: Time when ASI is achieved (or large value if not)
    """
    from backend.classes.us_project import US_SLOWDOWN_CAP_MARGIN
    original_dir = os.getcwd()

    try:
        takeoff_model_dir, base_time_series, present_day, TrajectoryPredictor, TakeoffParameters = _setup_takeoff_model()
        os.chdir(takeoff_model_dir)

        # Prepare compute arrays
        inference_to_use, experiment_to_use = _prepare_compute_arrays(
            covert_compute_years, covert_compute_values, base_time_series, present_day
        )

        # Convert capability_cap to numpy array and interpolate if needed
        capability_cap_array = None
        if capability_cap is not None:
            cap_array = np.array(capability_cap, dtype=float)
            # Apply margin to create visual separation between trajectory and cap
            if apply_cap_margin:
                cap_array = cap_array * US_SLOWDOWN_CAP_MARGIN
            if capability_cap_times is not None:
                # Interpolate to base_time_series.time
                times_array = np.array(capability_cap_times, dtype=float)
                capability_cap_array = np.interp(base_time_series.time, times_array, cap_array)
            elif len(cap_array) == len(base_time_series.time):
                capability_cap_array = cap_array
            else:
                print(f"Warning: capability_cap length ({len(cap_array)}) != base_time_series.time length ({len(base_time_series.time)})")
                return None

        # Run trajectory prediction with default parameters
        predictor = TrajectoryPredictor(params=TakeoffParameters())
        milestones_dict = predictor.predict_from_time_series(
            time=base_time_series.time,
            inference_compute=inference_to_use,
            experiment_compute=experiment_to_use,
            L_HUMAN=base_time_series.L_HUMAN,
            training_compute_growth_rate=base_time_series.training_compute_growth_rate,
            capability_cap=capability_cap_array
        )

        trajectory = predictor.get_full_trajectory()
        os.chdir(original_dir)

        if trajectory and 'times' in trajectory and 'ai_sw_progress_mult_ref_present_day' in trajectory:
            times = trajectory['times']
            speedup = trajectory['ai_sw_progress_mult_ref_present_day']
            # Smooth the speedup trajectory to remove bumps from discrete capability cap enforcement
            if isinstance(speedup, np.ndarray):
                speedup = _smooth_speedup_trajectory(speedup)
            else:
                speedup = _smooth_speedup_trajectory(np.array(speedup))

            # Extract ASI time
            asi_time = 1e308  # Large but finite value for JSON compatibility
            if 'ASI' in milestones_dict:
                asi_time = milestones_dict['ASI'].time

            result = {
                'trajectory_times': times.tolist() if hasattr(times, 'tolist') else list(times),
                'times': times.tolist() if hasattr(times, 'tolist') else list(times),  # Also include as 'times' for consistency
                'speedup': speedup.tolist() if hasattr(speedup, 'tolist') else list(speedup),
                'milestones': _serialize_milestones(milestones_dict),
                'asi_time': asi_time
            }

            # Include capped_compute if available (generated when capability cap is applied)
            if 'capped_compute' in trajectory:
                capped_compute = trajectory['capped_compute']
                result['capped_compute'] = capped_compute.tolist() if hasattr(capped_compute, 'tolist') else list(capped_compute)

            return result

        return None

    except Exception as e:
        print(f"Error in run_deterministic_trajectory: {e}")
        import traceback
        traceback.print_exc()
        try:
            os.chdir(original_dir)
        except:
            pass
        return None


def run_monte_carlo_trajectories(
    covert_compute_years: Optional[List[float]],
    covert_compute_values: Optional[List[float]],
    num_samples: int = DEFAULT_MC_SAMPLES,
    seed: int = 42,
    progress_callback: Optional[callable] = None,
    num_workers: Optional[int] = None
) -> Dict[str, Any]:
    """Run Monte Carlo simulations for uncertainty bands, with deterministic median.

    Uses parallel processing across multiple CPU cores for significant speedup.
    The median trajectory is computed using default parameters (not sampled).
    Only the p25/p75 uncertainty bands come from Monte Carlo sampling.

    Args:
        covert_compute_years: Years for the compute trajectory (or None for global)
        covert_compute_values: Compute values at each year (or None for global)
        num_samples: Number of Monte Carlo samples for uncertainty bands
        seed: Random seed for reproducibility
        progress_callback: Optional callback function(current, total) for progress updates
        num_workers: Number of parallel workers (defaults to cpu_count - 1)

    Returns:
        Dictionary containing:
            - trajectory_times: Common time grid for all trajectories
            - speedup_percentiles: Dict with 'p25', 'median', 'p75' arrays
            - milestones_median: Milestones from deterministic run
            - milestones_p25: Milestones from 25th percentile MC run
            - milestones_p75: Milestones from 75th percentile MC run
            - asi_times: Dict with ASI times for p25, median, p75
            - num_successful_samples: Number of successful MC samples
    """
    if num_workers is None:
        num_workers = DEFAULT_NUM_WORKERS

    original_dir = os.getcwd()

    try:
        takeoff_model_dir, base_time_series, present_day, TrajectoryPredictor, TakeoffParameters = _setup_takeoff_model()
        os.chdir(takeoff_model_dir)

        # Prepare compute arrays
        inference_to_use, experiment_to_use = _prepare_compute_arrays(
            covert_compute_years, covert_compute_values, base_time_series, present_day
        )

        # === 1. Run deterministic median trajectory with default parameters ===
        predictor = TrajectoryPredictor(params=TakeoffParameters())
        milestones_dict = predictor.predict_from_time_series(
            time=base_time_series.time,
            inference_compute=inference_to_use,
            experiment_compute=experiment_to_use,
            L_HUMAN=base_time_series.L_HUMAN,
            training_compute_growth_rate=base_time_series.training_compute_growth_rate
        )

        trajectory = predictor.get_full_trajectory()
        if not trajectory or 'times' not in trajectory or 'ai_sw_progress_mult_ref_present_day' not in trajectory:
            os.chdir(original_dir)
            return None

        common_times = trajectory['times']
        median_speedup = trajectory['ai_sw_progress_mult_ref_present_day']
        # Smooth the speedup trajectory to remove bumps from discrete capability cap enforcement
        if isinstance(median_speedup, np.ndarray):
            median_speedup = _smooth_speedup_trajectory(median_speedup)
        else:
            median_speedup = _smooth_speedup_trajectory(np.array(median_speedup))
        median_asi_time = 1e308
        if 'ASI' in milestones_dict:
            median_asi_time = milestones_dict['ASI'].time
        median_milestones = _serialize_milestones(milestones_dict)

        os.chdir(original_dir)

        # Report progress for deterministic run
        if progress_callback:
            progress_callback(1, num_samples + 1)

        # === 2. Run Monte Carlo for uncertainty bands in parallel ===
        mc_runs = []

        # Prepare arguments for parallel execution
        # Each sample gets a unique seed derived from base seed + index
        mc_args = [
            (i, seed + i, covert_compute_years, covert_compute_values)
            for i in range(num_samples)
        ]

        # Run in parallel using joblib (handles Flask/macOS edge cases better)
        if JOBLIB_AVAILABLE and num_workers > 1:
            # Use loky backend which handles forking issues on macOS
            print(f"Running {num_samples} MC samples in parallel with {num_workers} workers...", flush=True)
            results = Parallel(n_jobs=num_workers, backend='loky', verbose=0)(
                delayed(_run_single_mc_sample)(args) for args in mc_args
            )
            mc_runs = [r for r in results if r is not None]
            print(f"Completed {len(mc_runs)} successful MC samples", flush=True)

            # Report final progress
            if progress_callback:
                progress_callback(num_samples + 1, num_samples + 1)
        else:
            # Fall back to sequential execution
            for i, args in enumerate(mc_args):
                result = _run_single_mc_sample(args)
                if result is not None:
                    mc_runs.append(result)

                if progress_callback:
                    progress_callback(i + 2, num_samples + 1)

        # === 3. Extract p25 and p75 from MC runs ===
        if mc_runs:
            # Sort by ASI time to find percentile runs
            mc_runs_sorted = sorted(mc_runs, key=lambda x: x['asi_time'])
            n = len(mc_runs_sorted)

            idx_p25 = max(0, int(n * 0.25) - 1)
            idx_p75 = min(n - 1, int(n * 0.75))

            run_p25 = mc_runs_sorted[idx_p25]
            run_p75 = mc_runs_sorted[idx_p75]
        else:
            # Fallback: use median for all if MC failed
            run_p25 = {
                'speedup': median_speedup.tolist() if hasattr(median_speedup, 'tolist') else list(median_speedup),
                'milestones': median_milestones,
                'asi_time': median_asi_time
            }
            run_p75 = run_p25

        # Extract all individual MC run speedups for plotting
        all_mc_speedups = [run['speedup'] for run in mc_runs] if mc_runs else []

        return {
            'trajectory_times': common_times.tolist() if hasattr(common_times, 'tolist') else list(common_times),
            'speedup_percentiles': {
                'p25': run_p25['speedup'],
                'median': median_speedup.tolist() if hasattr(median_speedup, 'tolist') else list(median_speedup),
                'p75': run_p75['speedup']
            },
            'all_mc_runs': all_mc_speedups,  # All individual MC simulation trajectories
            'milestones_p25': run_p25['milestones'],
            'milestones_median': median_milestones,
            'milestones_p75': run_p75['milestones'],
            'asi_times': {
                'p25': run_p25['asi_time'],
                'median': median_asi_time,
                'p75': run_p75['asi_time']
            },
            'num_successful_samples': len(mc_runs) + 1  # +1 for deterministic
        }

    except Exception as e:
        print(f"Error in run_monte_carlo_trajectories: {e}")
        import traceback
        traceback.print_exc()
        try:
            os.chdir(original_dir)
        except:
            pass
        return None


def _run_mc_sample_with_trajectory_id(args: Tuple) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """Run a single MC sample and return with trajectory ID.

    Args:
        args: Tuple of (trajectory_id, sim_idx, seed, covert_compute_years, covert_compute_values)

    Returns:
        Tuple of (trajectory_id, job_type, result_dict or None)
        job_type is 'mc' for Monte Carlo sample
    """
    trajectory_id, sim_idx, seed, covert_compute_years, covert_compute_values = args
    result = _run_single_mc_sample((sim_idx, seed, covert_compute_years, covert_compute_values))
    return trajectory_id, 'mc', result


def _run_deterministic_with_trajectory_id(args: Tuple) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """Run a deterministic trajectory and return with trajectory ID.

    Args:
        args: Tuple of (trajectory_id, covert_compute_years, covert_compute_values)

    Returns:
        Tuple of (trajectory_id, job_type, result_dict or None)
        job_type is 'deterministic' for deterministic run
    """
    trajectory_id, covert_compute_years, covert_compute_values = args
    result = run_deterministic_trajectory(covert_compute_years, covert_compute_values)
    return trajectory_id, 'deterministic', result


def run_monte_carlo_batch_with_progress(
    trajectory_configs: List[Dict[str, Any]],
    num_samples: int = DEFAULT_MC_SAMPLES,
    base_seed: int = 42,
    progress_callback: Optional[callable] = None,
    num_workers: Optional[int] = None
) -> Dict[str, Dict[str, Any]]:
    """Run Monte Carlo simulations for multiple trajectories in a single parallel pool.

    This function runs all MC samples for all trajectories concurrently, reporting
    progress as each job completes. This maximizes CPU utilization and provides
    real-time progress updates.

    Args:
        trajectory_configs: List of trajectory configs, each containing:
            - 'name': Trajectory identifier (e.g., 'covert', 'proxy_project')
            - 'years': Compute years (or None for global)
            - 'values': Compute values (or None for global)
            - 'seed_offset': Offset to add to base_seed for this trajectory
        num_samples: Number of MC samples per trajectory
        base_seed: Base random seed
        progress_callback: Optional callback(completed, total, message) for progress
        num_workers: Number of parallel workers (defaults to all CPU cores)

    Returns:
        Dictionary mapping trajectory name to MC results
    """
    if num_workers is None:
        # Use ALL CPU cores for maximum parallelization
        num_workers = mp.cpu_count()

    # Build all jobs for all trajectories (MC samples + deterministic runs)
    mc_jobs = []
    det_jobs = []

    for config in trajectory_configs:
        traj_name = config['name']
        years = config.get('years')
        values = config.get('values')
        seed_offset = config.get('seed_offset', 0)

        # Add MC sample jobs
        for i in range(num_samples):
            job_seed = base_seed + seed_offset + i
            mc_jobs.append((traj_name, i, job_seed, years, values))

        # Add deterministic job for this trajectory
        det_jobs.append((traj_name, years, values))

    total_mc_jobs = len(mc_jobs)
    total_det_jobs = len(det_jobs)
    total_jobs = total_mc_jobs + total_det_jobs
    completed = [0]  # Use list to allow modification in closure

    # Results organized by trajectory
    results_by_trajectory = {config['name']: {'mc': [], 'deterministic': None} for config in trajectory_configs}

    if progress_callback:
        progress_callback(0, total_jobs, f"Starting {total_mc_jobs} MC + {total_det_jobs} deterministic simulations...")

    print(f"[MC] Running {total_jobs} total jobs ({total_mc_jobs} MC + {total_det_jobs} deterministic)", flush=True)
    print(f"[MC] Using {num_workers} workers (system has {mp.cpu_count()} CPU cores)", flush=True)

    # Timeout settings
    JOB_TIMEOUT = 30  # Per-job timeout (seconds)
    GLOBAL_TIMEOUT = 120  # Max time for all jobs (seconds)

    # Use ProcessPoolExecutor with spawn context for macOS compatibility
    # This gives us real-time progress via as_completed()
    mp_context = mp.get_context('spawn')
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    try:
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_context) as executor:
            # Submit all jobs to the executor (both MC and deterministic)
            future_to_job = {}

            # Submit MC jobs
            for job in mc_jobs:
                future = executor.submit(_run_mc_sample_with_trajectory_id, job)
                future_to_job[future] = ('mc', job)

            # Submit deterministic jobs
            for job in det_jobs:
                future = executor.submit(_run_deterministic_with_trajectory_id, job)
                future_to_job[future] = ('deterministic', job)

            print(f"[MC] Submitted all {total_jobs} jobs to executor pool", flush=True)

            # Process results as they complete with global timeout
            try:
                for future in as_completed(future_to_job, timeout=GLOBAL_TIMEOUT):
                    completed[0] += 1
                    job_type, job_args = future_to_job[future]

                    try:
                        traj_name, result_type, result = future.result(timeout=JOB_TIMEOUT)
                        if result is not None:
                            if result_type == 'mc':
                                results_by_trajectory[traj_name]['mc'].append(result)
                            else:  # deterministic
                                results_by_trajectory[traj_name]['deterministic'] = result
                    except FuturesTimeoutError:
                        print(f"[MC] Job timed out after {JOB_TIMEOUT}s", flush=True)
                    except Exception as e:
                        print(f"[MC] Job failed with error: {e}", flush=True)

                    # Report progress after each completion
                    if progress_callback:
                        progress_callback(completed[0], total_jobs, f"Simulations: {completed[0]}/{total_jobs}")

                    # Log every 10 completions for visibility
                    if completed[0] % 10 == 0 or completed[0] == total_jobs:
                        print(f"[MC] Progress: {completed[0]}/{total_jobs} ({100*completed[0]//total_jobs}%)", flush=True)

            except FuturesTimeoutError:
                print(f"[MC] Global timeout after {GLOBAL_TIMEOUT}s - canceling remaining jobs", flush=True)
                # Cancel any remaining futures
                for future in future_to_job:
                    future.cancel()

    except Exception as e:
        print(f"[MC] Executor error: {e}", flush=True)
        import traceback
        traceback.print_exc()

    print(f"[MC] Completed {completed[0]}/{total_jobs} simulations", flush=True)

    # Now process results for each trajectory
    final_results = {}

    for config in trajectory_configs:
        traj_name = config['name']
        mc_runs = results_by_trajectory[traj_name]['mc']
        det_result = results_by_trajectory[traj_name]['deterministic']

        if det_result is None:
            print(f"[MC] WARNING: Deterministic trajectory for {traj_name} failed!", flush=True)
            continue

        common_times = det_result['trajectory_times']
        median_speedup = det_result['speedup']
        median_milestones = det_result['milestones']
        median_asi_time = det_result['asi_time']

        # Extract p25 and p75 from MC runs
        if mc_runs:
            mc_runs_sorted = sorted(mc_runs, key=lambda x: x['asi_time'])
            n = len(mc_runs_sorted)
            idx_p25 = max(0, int(n * 0.25) - 1)
            idx_p75 = min(n - 1, int(n * 0.75))
            run_p25 = mc_runs_sorted[idx_p25]
            run_p75 = mc_runs_sorted[idx_p75]
        else:
            run_p25 = {'speedup': median_speedup, 'milestones': median_milestones, 'asi_time': median_asi_time}
            run_p75 = run_p25

        all_mc_speedups = [run['speedup'] for run in mc_runs] if mc_runs else []

        final_results[traj_name] = {
            'trajectory_times': common_times,
            'speedup_percentiles': {
                'p25': run_p25['speedup'],
                'median': median_speedup,
                'p75': run_p75['speedup']
            },
            'all_mc_runs': all_mc_speedups,
            'milestones_p25': run_p25['milestones'],
            'milestones_median': median_milestones,
            'milestones_p75': run_p75['milestones'],
            'asi_times': {
                'p25': run_p25['asi_time'],
                'median': median_asi_time,
                'p75': run_p75['asi_time']
            },
            'num_successful_samples': len(mc_runs) + 1
        }

    return final_results
