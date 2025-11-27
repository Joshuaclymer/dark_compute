"""
P(catastrophe) calculations for the slowdown model.

This module handles computing P(catastrophe) from trajectory milestones
and generating curve data for plotting.
"""

import numpy as np
from typing import Dict, Any, List

from backend.paramaters import PCatastropheParameters
from backend.classes.p_catastrophe import PCatastrophe


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
