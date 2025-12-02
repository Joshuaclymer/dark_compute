"""
Trajectory computation for AI R&D speedup analysis.

This module handles extracting compute trajectories and predicting
AI capability milestones using the takeoff model.
"""

import numpy as np
import os
import sys
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.paramaters import ModelParameters


def extract_takeoff_slowdown_trajectories(
    app_params: 'ModelParameters',
    covert_compute_years: Optional[List[float]] = None,
    covert_compute_median: Optional[List[float]] = None
) -> Dict[str, Any]:
    """Extract compute trajectories for AI takeoff slowdown analysis.

    Returns trajectory data including:
    1. AI R&D speedup time series
    2. Milestones (AC, SAR, SIAR, TED-AI, ASI) with times and speedup values

    Uses the default takeoff model parameters and compute trajectory from
    the AI Futures Project.

    Args:
        app_params: Application parameters (for agreement year only)
        covert_compute_years: Optional list of years for covert compute trajectory
        covert_compute_median: Optional list of covert compute values (H100e)

    Returns:
        Dictionary with trajectory arrays and milestone information
    """
    # Define time range from agreement year forward
    agreement_year = app_params.simulation_settings.start_agreement_at_specific_year

    # Predict milestones using takeoff model with its default parameters
    milestones_global = None
    trajectory_global = None

    try:
        # Save current directory
        original_dir = os.getcwd()
        # Change to takeoff model directory
        takeoff_model_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'takeoff_model', 'ai-futures-calculator'
        )
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
            original_dir = os.getcwd()
            takeoff_model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'takeoff_model', 'ai-futures-calculator'
            )
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
            # identical between global and PRC trajectories.
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

            print(f"Modified trajectory: inference at 2030 = {modified_inference[base_time_series.time >= 2030][0]:.2e}")
            print(f"Modified trajectory: experiment at 2030 = {modified_experiment[base_time_series.time >= 2030][0]:.2e}")

            # Debug: Verify present_day compute is UNCHANGED
            print(f"DEBUG: Modified inference at present_day: {modified_inference[present_day_idx]:.2e}")
            print(f"DEBUG: Modified experiment at present_day: {modified_experiment[present_day_idx]:.2e}")

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
