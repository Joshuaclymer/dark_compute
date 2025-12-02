"""
Serve data for the AI takeoff slowdown model visualization.

This package computes AI R&D speedup trajectories using the takeoff model
to predict when key capability milestones will be reached.
"""

from .main import (
    get_slowdown_model_data,
    get_slowdown_model_data_with_progress,
    get_trajectory_data_fast,
    get_uncertainty_data,
)

__all__ = [
    'get_slowdown_model_data',
    'get_slowdown_model_data_with_progress',
    'get_trajectory_data_fast',
    'get_uncertainty_data',
]
