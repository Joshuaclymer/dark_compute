
# import numpy as np
# import os
# import sys
# from typing import Dict, Any, List


# def _get_takeoff_model_dir():
#     """Get the path to the takeoff model directory."""
#     return os.path.join(
#         os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
#         'takeoff_model', 'ai-futures-calculator'
#     )


# def _setup_takeoff_model():
#     """Set up the takeoff model environment and return necessary imports and data.

#     Returns:
#         Tuple of (takeoff_model_dir, base_time_series, present_day, TrajectoryPredictor, TakeoffParameters)
#     """
#     takeoff_model_dir = _get_takeoff_model_dir()

#     if takeoff_model_dir not in sys.path:
#         sys.path.insert(0, takeoff_model_dir)
#     scripts_dir = os.path.join(takeoff_model_dir, 'scripts')
#     if scripts_dir not in sys.path:
#         sys.path.insert(0, scripts_dir)

#     from predict_trajectory import TrajectoryPredictor
#     from progress_model import load_time_series_data, Parameters as TakeoffParameters

#     # Load base time series data
#     default_csv = os.path.join(takeoff_model_dir, 'input_data.csv')
#     base_time_series = load_time_series_data(default_csv)
#     present_day = TakeoffParameters().present_day

#     return takeoff_model_dir, base_time_series, present_day, TrajectoryPredictor, TakeoffParameters


# class TakeoffModel:
#     """
#     Takeoff model with Monte Carlo sampled parameters.

#     At initialization, samples parameters from the takeoff model's parameter
#     distributions and stores them for later trajectory predictions.

#     Attributes:
#         sampled_params_list: List of sampled parameter dictionaries
#         num_samples: Number of Monte Carlo samples
#     """

#     def __init__(self, num_samples: int = 50, seed: int = 42):
#         """
#         Initialize the TakeoffModel with Monte Carlo sampled parameters.

#         Args:
#             num_samples: Number of Monte Carlo samples to draw
#             seed: Random seed for reproducibility
#         """
#         self.num_samples = num_samples
#         self.seed = seed
#         self.sampled_params_list: List[Dict[str, Any]] = []

#         # Sample parameters at initialization
#         self._sample_parameters()

#     def _sample_parameters(self):
#         """Sample parameters from the takeoff model's distribution configuration."""
#         takeoff_model_dir = _get_takeoff_model_dir()

#         if takeoff_model_dir not in sys.path:
#             sys.path.insert(0, takeoff_model_dir)
#         scripts_dir = os.path.join(takeoff_model_dir, 'scripts')
#         if scripts_dir not in sys.path:
#             sys.path.insert(0, scripts_dir)

#         from batch_rollout import _load_config, _sample_from_dist, _clip_to_param_bounds

#         # Load sampling configuration
#         sampling_config_path = os.path.join(takeoff_model_dir, 'config', 'sampling_config.yaml')
#         sampling_config = _load_config(sampling_config_path)
#         param_dists = sampling_config.get('parameters', {})

#         rng = np.random.default_rng(self.seed)

#         for _ in range(self.num_samples):
#             # Sample parameters from distributions
#             sampled_params = {}
#             for name, spec in param_dists.items():
#                 if name == "automation_anchors":
#                     continue
#                 val = _sample_from_dist(spec, rng, name)
#                 clip_requested = spec.get("clip_to_bounds", True)
#                 if clip_requested:
#                     val = _clip_to_param_bounds(name, val)
#                 sampled_params[name] = val

#             self.sampled_params_list.append(sampled_params)

#     def get_agreement_start_year(self, start_agreement_at_what_ai_rnd_speedup: float) -> Dict[str, float]:
#         """
#         Determine when an agreement should start based on AI R&D speedup threshold.

#         Runs trajectories with default compute and finds the year when AI R&D speedup
#         reaches the specified threshold for each sampled parameter set.

#         Args:
#             start_agreement_at_what_ai_rnd_speedup: The AI R&D speedup multiplier
#                 at which the agreement should start (e.g., 2.0 for 2x speedup)

#         Returns:
#             Dictionary with:
#                 - 'median': Median year across all samples
#                 - 'p25': 25th percentile year
#                 - 'p75': 75th percentile year
#                 - 'years': List of all years for each sample
#         """
#         original_dir = os.getcwd()

#         try:
#             takeoff_model_dir, base_time_series, _, TrajectoryPredictor, TakeoffParameters = _setup_takeoff_model()
#             os.chdir(takeoff_model_dir)

#             agreement_years = []

#             for sampled_params in self.sampled_params_list:
#                 try:
#                     predictor = TrajectoryPredictor(params=TakeoffParameters(**sampled_params))
#                     predictor.predict_from_time_series(
#                         time=base_time_series.time,
#                         inference_compute=base_time_series.inference_compute,
#                         experiment_compute=base_time_series.experiment_compute,
#                         L_HUMAN=base_time_series.L_HUMAN,
#                         training_compute_growth_rate=base_time_series.training_compute_growth_rate
#                     )

#                     trajectory = predictor.get_full_trajectory()

#                     if trajectory and 'times' in trajectory and 'ai_sw_progress_mult_ref_present_day' in trajectory:
#                         times = np.array(trajectory['times'])
#                         speedup = np.array(trajectory['ai_sw_progress_mult_ref_present_day'])

#                         # Find the first year where speedup exceeds threshold
#                         indices = np.where(speedup >= start_agreement_at_what_ai_rnd_speedup)[0]
#                         if len(indices) > 0:
#                             agreement_year = times[indices[0]]
#                             agreement_years.append(float(agreement_year))
#                         else:
#                             # Speedup never reached - use end of trajectory
#                             agreement_years.append(float(times[-1]))
#                 except Exception as e:
#                     print(f"Warning: Failed to compute trajectory for sample: {e}")
#                     continue

#             os.chdir(original_dir)

#             if not agreement_years:
#                 return {
#                     'median': None,
#                     'p25': None,
#                     'p75': None,
#                     'years': []
#                 }

#             agreement_years = np.array(agreement_years)

#             return {
#                 'median': float(np.median(agreement_years)),
#                 'p25': float(np.percentile(agreement_years, 25)),
#                 'p75': float(np.percentile(agreement_years, 75)),
#                 'years': agreement_years.tolist()
#             }

#         except Exception as e:
#             print(f"Error in get_agreement_start_year: {e}")
#             import traceback
#             traceback.print_exc()
#             try:
#                 os.chdir(original_dir)
#             except:
#                 pass
#             return {
#                 'median': None,
#                 'p25': None,
#                 'p75': None,
#                 'years': []
#             }

#     def predict_trajectory(
#         self,
#         time: List[float],
#         human_labor: List[float],
#         compute: List[float],
#         proportion_compute_spent_on_experiments: float = 0.15
#     ) -> Dict[str, Any]:
#         """
#         Predict AI R&D speedup trajectory given custom inputs.

#         Runs the takeoff model with each sampled parameter set using the provided
#         time series inputs, and returns percentile statistics across all samples.

#         Args:
#             time: List of time points (decimal years)
#             human_labor: List of human labor values at each time point
#             compute: List of total compute values at each time point (H100-equivalents)
#             proportion_compute_spent_on_experiments: Fraction of compute for experiments
#                 (default 0.15 means 15% for inference, 85% for experiments)

#         Returns:
#             Dictionary containing:
#                 - 'trajectory_times': Time grid for trajectories
#                 - 'speedup_percentiles': Dict with 'p25', 'median', 'p75' arrays
#                 - 'all_speedups': List of all individual speedup trajectories
#                 - 'milestones_median': Milestones from median trajectory
#                 - 'asi_times': Dict with ASI times for p25, median, p75
#                 - 'num_successful_samples': Number of successful trajectory computations
#         """
#         original_dir = os.getcwd()

#         try:
#             takeoff_model_dir, base_time_series, _, TrajectoryPredictor, TakeoffParameters = _setup_takeoff_model()
#             os.chdir(takeoff_model_dir)

#             # Convert inputs to numpy arrays
#             time_array = np.array(time, dtype=float)
#             human_labor_array = np.array(human_labor, dtype=float)
#             compute_array = np.array(compute, dtype=float)

#             # Split compute into inference and experiment
#             inference_compute = compute_array * (1 - proportion_compute_spent_on_experiments)
#             experiment_compute = compute_array * proportion_compute_spent_on_experiments

#             # Use training compute growth rate from base time series, interpolated to our time grid
#             training_compute_growth_rate = np.interp(
#                 time_array,
#                 base_time_series.time,
#                 base_time_series.training_compute_growth_rate
#             )

#             successful_runs = []

#             for sampled_params in self.sampled_params_list:
#                 try:
#                     predictor = TrajectoryPredictor(params=TakeoffParameters(**sampled_params))
#                     milestones_dict = predictor.predict_from_time_series(
#                         time=time_array,
#                         inference_compute=inference_compute,
#                         experiment_compute=experiment_compute,
#                         L_HUMAN=human_labor_array,
#                         training_compute_growth_rate=training_compute_growth_rate
#                     )

#                     trajectory = predictor.get_full_trajectory()

#                     if trajectory and 'times' in trajectory and 'ai_sw_progress_mult_ref_present_day' in trajectory:
#                         speedup = trajectory['ai_sw_progress_mult_ref_present_day']

#                         # Extract ASI time
#                         asi_time = 1e308
#                         if 'ASI' in milestones_dict:
#                             asi_time = milestones_dict['ASI'].time

#                         # Serialize milestones
#                         milestones_serialized = {
#                             name: {
#                                 'time': float(info.time),
#                                 'progress_level': float(info.progress_level),
#                                 'progress_multiplier': float(info.progress_multiplier)
#                             }
#                             for name, info in milestones_dict.items()
#                         }

#                         successful_runs.append({
#                             'speedup': speedup.tolist() if hasattr(speedup, 'tolist') else list(speedup),
#                             'milestones': milestones_serialized,
#                             'asi_time': asi_time,
#                             'params': sampled_params
#                         })

#                 except Exception as e:
#                     print(f"Warning: Failed trajectory computation: {e}")
#                     continue

#             os.chdir(original_dir)

#             if not successful_runs:
#                 return {
#                     'trajectory_times': time,
#                     'speedup_percentiles': {'p25': [], 'median': [], 'p75': []},
#                     'all_speedups': [],
#                     'milestones_median': {},
#                     'asi_times': {'p25': None, 'median': None, 'p75': None},
#                     'num_successful_samples': 0
#                 }

#             # Sort by ASI time to find percentile runs
#             runs_sorted = sorted(successful_runs, key=lambda x: x['asi_time'])
#             n = len(runs_sorted)

#             idx_p25 = max(0, int(n * 0.25) - 1)
#             idx_median = n // 2
#             idx_p75 = min(n - 1, int(n * 0.75))

#             run_p25 = runs_sorted[idx_p25]
#             run_median = runs_sorted[idx_median]
#             run_p75 = runs_sorted[idx_p75]

#             # Extract all speedups
#             all_speedups = [run['speedup'] for run in successful_runs]

#             return {
#                 'trajectory_times': time,
#                 'speedup_percentiles': {
#                     'p25': run_p25['speedup'],
#                     'median': run_median['speedup'],
#                     'p75': run_p75['speedup']
#                 },
#                 'all_speedups': all_speedups,
#                 'milestones_p25': run_p25['milestones'],
#                 'milestones_median': run_median['milestones'],
#                 'milestones_p75': run_p75['milestones'],
#                 'asi_times': {
#                     'p25': run_p25['asi_time'],
#                     'median': run_median['asi_time'],
#                     'p75': run_p75['asi_time']
#                 },
#                 'num_successful_samples': len(successful_runs)
#             }

#         except Exception as e:
#             print(f"Error in predict_trajectory: {e}")
#             import traceback
#             traceback.print_exc()
#             try:
#                 os.chdir(original_dir)
#             except:
#                 pass
#             return {
#                 'trajectory_times': time,
#                 'speedup_percentiles': {'p25': [], 'median': [], 'p75': []},
#                 'all_speedups': [],
#                 'milestones_median': {},
#                 'asi_times': {'p25': None, 'median': None, 'p75': None},
#                 'num_successful_samples': 0
#             }

#     def predict_trajectory_deterministic(
#         self,
#         time: List[float],
#         human_labor: List[float],
#         compute: List[float],
#         proportion_compute_spent_on_experiments: float = 0.15,
#         capability_cap: List[float] = None
#     ) -> Dict[str, Any]:
#         """
#         Predict AI R&D speedup trajectory using default (median) parameters only.

#         This is faster than predict_trajectory as it only runs a single simulation
#         with default parameters rather than Monte Carlo sampling.

#         Args:
#             time: List of time points (decimal years)
#             human_labor: List of human labor values at each time point
#             compute: List of total compute values at each time point (H100-equivalents)
#             proportion_compute_spent_on_experiments: Fraction of compute for experiments
#             capability_cap: Optional list of progress cap values at each time point.
#                 If provided, progress will be upper-bounded by these values.

#         Returns:
#             Dictionary containing:
#                 - 'trajectory_times': Time points
#                 - 'speedup': AI R&D speedup values
#                 - 'milestones': Milestone information
#                 - 'asi_time': Time when ASI is achieved
#         """
#         original_dir = os.getcwd()

#         try:
#             takeoff_model_dir, base_time_series, _, TrajectoryPredictor, TakeoffParameters = _setup_takeoff_model()
#             os.chdir(takeoff_model_dir)

#             # Convert inputs to numpy arrays
#             time_array = np.array(time, dtype=float)
#             human_labor_array = np.array(human_labor, dtype=float)
#             compute_array = np.array(compute, dtype=float)

#             # Split compute into inference and experiment
#             inference_compute = compute_array * (1 - proportion_compute_spent_on_experiments)
#             experiment_compute = compute_array * proportion_compute_spent_on_experiments

#             # Use training compute growth rate from base time series
#             training_compute_growth_rate = np.interp(
#                 time_array,
#                 base_time_series.time,
#                 base_time_series.training_compute_growth_rate
#             )

#             # Convert capability_cap to numpy array if provided
#             capability_cap_array = None
#             if capability_cap is not None:
#                 capability_cap_array = np.array(capability_cap, dtype=float)

#             # Run with default parameters
#             predictor = TrajectoryPredictor(params=TakeoffParameters())
#             milestones_dict = predictor.predict_from_time_series(
#                 time=time_array,
#                 inference_compute=inference_compute,
#                 experiment_compute=experiment_compute,
#                 L_HUMAN=human_labor_array,
#                 training_compute_growth_rate=training_compute_growth_rate,
#                 capability_cap=capability_cap_array
#             )

#             trajectory = predictor.get_full_trajectory()
#             os.chdir(original_dir)

#             if trajectory and 'times' in trajectory and 'ai_sw_progress_mult_ref_present_day' in trajectory:
#                 speedup = trajectory['ai_sw_progress_mult_ref_present_day']

#                 asi_time = 1e308
#                 if 'ASI' in milestones_dict:
#                     asi_time = milestones_dict['ASI'].time

#                 milestones_serialized = {
#                     name: {
#                         'time': float(info.time),
#                         'progress_level': float(info.progress_level),
#                         'progress_multiplier': float(info.progress_multiplier)
#                     }
#                     for name, info in milestones_dict.items()
#                 }

#                 return {
#                     'trajectory_times': time,
#                     'speedup': speedup.tolist() if hasattr(speedup, 'tolist') else list(speedup),
#                     'milestones': milestones_serialized,
#                     'asi_time': asi_time
#                 }

#             return None

#         except Exception as e:
#             print(f"Error in predict_trajectory_deterministic: {e}")
#             import traceback
#             traceback.print_exc()
#             try:
#                 os.chdir(original_dir)
#             except:
#                 pass
#             return None