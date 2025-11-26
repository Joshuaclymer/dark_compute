"""
Serve data for the AI takeoff slowdown model visualization.

This module computes AI R&D speedup trajectories using the takeoff model
to predict when key capability milestones will be reached.
"""

import numpy as np
from typing import Dict, Any


def extract_takeoff_slowdown_trajectories(app_params: 'Parameters') -> Dict[str, Any]:
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

    return {
        'milestones_global': milestones_global,
        'trajectory_times': trajectory_times,
        'global_ai_speedup': global_ai_speedup
    }
