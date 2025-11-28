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
    end_time: float,
    safety_speedup_exponent: float = 1.0
) -> float:
    """Compute the safety-research-speed-adjusted duration between two time points.

    This integrates the safety research speedup over the handoff period to get the
    effective research time (accounting for accelerated progress).

    Safety speedup = capability_speedup ^ safety_speedup_exponent
    e.g., if exponent=0.5, safety speedup is sqrt of capability speedup

    Args:
        trajectory_times: Time points of the trajectory
        ai_speedup: AI R&D (capability) speedup values at each time point
        start_time: Start time (e.g., AC milestone time)
        end_time: End time (e.g., SAR milestone time)
        safety_speedup_exponent: Exponent relating safety to capability speedup (default 1.0 = same as capability)

    Returns:
        Safety-research-speed-adjusted duration in years
    """
    times = np.array(trajectory_times)
    capability_speedup = np.array(ai_speedup)

    # Convert capability speedup to safety speedup
    safety_speedup = np.power(np.maximum(capability_speedup, 1e-10), safety_speedup_exponent)

    # Find indices within the time range
    mask = (times >= start_time) & (times <= end_time)
    if not np.any(mask):
        return end_time - start_time  # Fallback to raw duration

    relevant_times = times[mask]
    relevant_speedup = safety_speedup[mask]

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

            # Safety-research-speed-adjusted handoff duration
            adjusted_handoff_duration = compute_research_speed_adjusted_duration(
                trajectory_times, global_ai_speedup, ac_time, sar_time,
                safety_speedup_exponent=p_cat_params.safety_speedup_exponent
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
                covert_trajectory_times, covert_ai_speedup, ac_time, sar_time,
                safety_speedup_exponent=p_cat_params.safety_speedup_exponent
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
    # Use log-spaced time points from ANCHOR_T1 to 20 years for log-scale x-axis
    # Start at ANCHOR_T1 since log(0) is undefined
    durations = np.logspace(np.log10(PCatastrophe.ANCHOR_T1), np.log10(20), 100)

    p_takeover = [p_cat.p_AI_takeover(d, p_cat_params) for d in durations]
    p_power_grabs = [p_cat.p_human_power_grabs(d, p_cat_params) for d in durations]

    return {
        'durations': durations.tolist(),
        'p_ai_takeover': p_takeover,
        'p_human_power_grabs': p_power_grabs,
        # Include the anchor points for reference
        'anchor_points': {
            'ai_takeover': {
                't1': p_cat_params.p_ai_takeover_t1,
                't2': p_cat_params.p_ai_takeover_t2,
                't3': p_cat_params.p_ai_takeover_t3
            },
            'human_power_grabs': {
                't1': p_cat_params.p_human_power_grabs_t1,
                't2': p_cat_params.p_human_power_grabs_t2,
                't3': p_cat_params.p_human_power_grabs_t3
            }
        },
        'anchor_times': {
            't1': PCatastrophe.ANCHOR_T1,
            't2': PCatastrophe.ANCHOR_T2,
            't3': PCatastrophe.ANCHOR_T3
        }
    }


def compute_p_catastrophe_over_time(
    mc_data: Dict[str, Any],
    agreement_year: float,
    p_cat_params: PCatastropheParameters
) -> Dict[str, Any]:
    """Compute P(catastrophe) over time during the slowdown period.

    This computes the probability of catastrophe at each time point after the agreement,
    where:
    - Handoff duration (so far) = current time - SC milestone time
    - P(human power grabs) uses raw handoff duration
    - P(AI takeover) uses research-speed-adjusted handoff duration
    - P(catastrophe) = 1 - (1 - P(takeover)) * (1 - P(power grabs))

    Args:
        mc_data: Monte Carlo results containing trajectory_times, speedup_percentiles, and milestones
                 This should be the proxy_project (US Frontier) trajectory data
        agreement_year: The year when the slowdown agreement starts
        p_cat_params: PCatastropheParameters with probability anchor points

    Returns:
        Dictionary containing:
            - years: Time points (years)
            - slowdown_duration: Duration of slowdown at each time point
            - handoff_duration: Handoff duration (time since SC) at each time point
            - adjusted_handoff_duration: Research-speed-adjusted handoff duration at each time point
            - p_ai_takeover: P(AI takeover) at each time point
            - p_human_power_grabs: P(human power grabs) at each time point
            - p_catastrophe: Combined P(catastrophe) at each time point
            - sc_time: Time when SC milestone is reached
    """
    if not mc_data:
        return None

    trajectory_times = mc_data.get('trajectory_times', [])
    speedup_percentiles = mc_data.get('speedup_percentiles', {})
    milestones = mc_data.get('milestones_median', {})

    if not trajectory_times or not speedup_percentiles or not milestones:
        return None

    median_speedup = speedup_percentiles.get('median', [])
    if not median_speedup:
        return None

    # Get SC (Superhuman Coder / AC) milestone time from the US Frontier trajectory
    sc_milestone = milestones.get('AC', {})
    sc_time = sc_milestone.get('time') if sc_milestone else None

    if sc_time is None:
        # If no SC milestone, we can't compute handoff duration
        return None

    p_cat = PCatastrophe()

    times = np.array(trajectory_times)
    speedup = np.array(median_speedup)

    # Filter to times after the agreement year (or after SC if SC is after agreement)
    start_time = max(agreement_year, sc_time)

    # Find indices for times >= start_time
    valid_indices = np.where(times >= start_time)[0]

    if len(valid_indices) == 0:
        return None

    result_years = []
    result_slowdown_duration = []
    result_handoff_duration = []
    result_adjusted_handoff = []
    result_p_takeover = []
    result_p_power_grabs = []
    result_p_catastrophe = []

    for idx in valid_indices:
        current_time = times[idx]

        # Duration of slowdown = current time - agreement year
        slowdown_duration = current_time - agreement_year

        # Handoff duration (so far) = current time - SC time
        handoff_duration = current_time - sc_time

        # Skip if handoff duration is <= 0
        if handoff_duration <= 0:
            continue

        # Compute safety-research-speed-adjusted handoff duration
        # Integrate speedup from SC time to current time
        adjusted_handoff = compute_research_speed_adjusted_duration(
            trajectory_times, median_speedup, sc_time, current_time,
            safety_speedup_exponent=p_cat_params.safety_speedup_exponent
        )

        # Compute probabilities using the PCatastrophe class
        p_takeover = p_cat.p_AI_takeover(adjusted_handoff, p_cat_params)
        p_power_grabs = p_cat.p_human_power_grabs(handoff_duration, p_cat_params)
        p_catastrophe = p_cat.p_domestic_takeover(handoff_duration, adjusted_handoff, p_cat_params)

        result_years.append(float(current_time))
        result_slowdown_duration.append(float(slowdown_duration))
        result_handoff_duration.append(float(handoff_duration))
        result_adjusted_handoff.append(float(adjusted_handoff))
        result_p_takeover.append(float(p_takeover))
        result_p_power_grabs.append(float(p_power_grabs))
        result_p_catastrophe.append(float(p_catastrophe))

    return {
        'years': result_years,
        'slowdown_duration': result_slowdown_duration,
        'handoff_duration': result_handoff_duration,
        'adjusted_handoff_duration': result_adjusted_handoff,
        'p_ai_takeover': result_p_takeover,
        'p_human_power_grabs': result_p_power_grabs,
        'p_catastrophe': result_p_catastrophe,
        'sc_time': float(sc_time),
        'agreement_year': float(agreement_year)
    }


def compute_optimal_compute_cap_over_time(
    p_catastrophe_over_time: Dict[str, Any],
    covert_compute_data: Dict[str, Any],
    agreement_year: float
) -> Dict[str, Any]:
    """Compute the optimal compute cap over time based on P(catastrophe).

    At each time point, the optimal compute cap is defined as the value at the
    (1 - P_catastrophe) quantile of the PRC covert compute distribution.

    For example, if P_catastrophe = 0.2 at time t, then the optimal cap is the
    compute value at the 80th percentile (0.8 quantile) of the covert distribution.

    Args:
        p_catastrophe_over_time: Result from compute_p_catastrophe_over_time()
        covert_compute_data: Dict containing 'years' and 'operational_dark_compute'
            where operational_dark_compute has keys like 'p10', 'p25', 'median', 'p75', 'p90'
        agreement_year: The year when the slowdown agreement starts

    Returns:
        Dictionary containing:
            - years: Time points (years)
            - slowdown_duration: Duration of slowdown at each time point
            - optimal_quantile: The quantile (1 - P_catastrophe) at each time point
            - optimal_compute_cap: The compute value at that quantile
            - p_catastrophe: P(catastrophe) at each time point (for reference)
    """
    if not p_catastrophe_over_time or not covert_compute_data:
        return None

    p_cat_years = p_catastrophe_over_time.get('years', [])
    p_cat_values = p_catastrophe_over_time.get('p_catastrophe', [])
    slowdown_durations = p_catastrophe_over_time.get('slowdown_duration', [])

    if not p_cat_years or not p_cat_values:
        return None

    # Get covert compute distribution data
    covert_years = covert_compute_data.get('years', [])
    operational = covert_compute_data.get('operational_dark_compute', {})

    if not covert_years or not operational:
        return None

    # Build percentile arrays for interpolation
    # Available percentiles: p10, p25, median (p50), p75, p90
    percentile_keys = ['p10', 'p25', 'median', 'p75', 'p90']
    percentile_values = [0.10, 0.25, 0.50, 0.75, 0.90]

    # Check which percentiles are available
    available_percentiles = []
    available_keys = []
    for pval, pkey in zip(percentile_values, percentile_keys):
        if pkey in operational and operational[pkey]:
            available_percentiles.append(pval)
            available_keys.append(pkey)

    if len(available_percentiles) < 2:
        return None

    covert_years_arr = np.array(covert_years)

    result_years = []
    result_slowdown_duration = []
    result_optimal_quantile = []
    result_optimal_compute_cap = []
    result_p_catastrophe = []

    for i, year in enumerate(p_cat_years):
        p_cat = p_cat_values[i]
        slowdown_dur = slowdown_durations[i]

        # Optimal quantile = 1 - P(catastrophe)
        optimal_quantile = 1.0 - p_cat

        # Clamp to available percentile range
        optimal_quantile_clamped = max(min(available_percentiles), min(max(available_percentiles), optimal_quantile))

        # Find the compute value at this year by interpolating the covert distribution
        # First, get the compute values at each available percentile for this year
        compute_at_percentiles = []
        for pkey in available_keys:
            percentile_data = operational[pkey]
            # Interpolate to get the value at the current year
            if year >= covert_years_arr[0] and year <= covert_years_arr[-1]:
                compute_val = float(np.interp(year, covert_years_arr, percentile_data))
            elif year < covert_years_arr[0]:
                compute_val = percentile_data[0]
            else:
                compute_val = percentile_data[-1]
            compute_at_percentiles.append(compute_val)

        # Now interpolate between percentiles to get the compute at the optimal quantile
        optimal_compute = float(np.interp(optimal_quantile_clamped, available_percentiles, compute_at_percentiles))

        result_years.append(float(year))
        result_slowdown_duration.append(float(slowdown_dur))
        result_optimal_quantile.append(float(optimal_quantile))
        result_optimal_compute_cap.append(float(optimal_compute))
        result_p_catastrophe.append(float(p_cat))

    return {
        'years': result_years,
        'slowdown_duration': result_slowdown_duration,
        'optimal_quantile': result_optimal_quantile,
        'optimal_compute_cap': result_optimal_compute_cap,
        'p_catastrophe': result_p_catastrophe,
        'agreement_year': float(agreement_year)
    }
