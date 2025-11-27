"""
Monte Carlo simulations for trajectory uncertainty.

This module runs Monte Carlo simulations of trajectory predictions
with sampled parameters to quantify uncertainty in AI R&D speedup curves.
"""

import numpy as np
import os
import sys
from typing import Dict, Any, Optional, List

# Default number of Monte Carlo samples for trajectory uncertainty
DEFAULT_MC_SAMPLES = 50


def run_monte_carlo_trajectories(
    covert_compute_years: Optional[List[float]],
    covert_compute_values: Optional[List[float]],
    num_samples: int = DEFAULT_MC_SAMPLES,
    seed: int = 42,
    progress_callback: Optional[callable] = None
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
    original_dir = os.getcwd()
    takeoff_model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
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
                milestones_dict = predictor.predict_from_time_series(
                    time=base_time_series.time,
                    inference_compute=inference_to_use,
                    experiment_compute=experiment_to_use,
                    L_HUMAN=base_time_series.L_HUMAN,
                    training_compute_growth_rate=base_time_series.training_compute_growth_rate
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
        print(f"Error in run_monte_carlo_trajectories: {e}")
        import traceback
        traceback.print_exc()
        try:
            os.chdir(original_dir)
        except:
            pass
        return None
