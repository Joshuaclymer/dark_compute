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
    safety_speedup_multiplier: float = 1.0,
    max_alignment_speedup: float = None
) -> float:
    """Compute the safety-research-speed-adjusted duration between two time points.

    This integrates the safety research speedup over the handoff period to get the
    effective research time (accounting for accelerated progress).

    Safety speedup = min(max_alignment_speedup, capability_speedup * safety_speedup_multiplier)
    e.g., if multiplier=0.5 and max=5, safety speedup is half of capability speedup, capped at 5

    Args:
        trajectory_times: Time points of the trajectory
        ai_speedup: AI R&D (capability) speedup values at each time point
        start_time: Start time (e.g., AC milestone time)
        end_time: End time (e.g., SAR milestone time)
        safety_speedup_multiplier: Multiplier relating safety to capability speedup (default 1.0 = same as capability)
        max_alignment_speedup: Maximum alignment speedup cap (default None = no cap)

    Returns:
        Safety-research-speed-adjusted duration in years
    """
    times = np.array(trajectory_times)
    capability_speedup = np.array(ai_speedup)

    # Convert capability speedup to safety speedup
    safety_speedup = capability_speedup * safety_speedup_multiplier

    # Apply cap if specified
    if max_alignment_speedup is not None:
        safety_speedup = np.minimum(safety_speedup, max_alignment_speedup)

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
        # AC = Automated Coder, SAR = Strong AI Researcher, ASI = Artificial Superintelligence
        # Handoff duration (for AI takeover) is the time between AC and SAR
        # Human power grabs risk uses SAR to ASI duration
        ac_time = milestones_global.get('AC', {}).get('time')
        sar_time = milestones_global.get('SAR-level-experiment-selection-skill', {}).get('time')
        asi_time = milestones_global.get('ASI', {}).get('time')

        if ac_time and sar_time:
            # Handoff duration: AC to SAR (for AI takeover calculation)
            handoff_duration = sar_time - ac_time

            # Safety-research-speed-adjusted handoff duration
            adjusted_handoff_duration = compute_research_speed_adjusted_duration(
                trajectory_times, global_ai_speedup, ac_time, sar_time,
                safety_speedup_multiplier=p_cat_params.safety_speedup_multiplier,
                max_alignment_speedup=p_cat_params.max_alignment_speedup_before_handoff
            )

            p_takeover = p_cat.p_AI_takeover(adjusted_handoff_duration, p_cat_params)

            # For human power grabs, use the trajectory with milestones
            trajectory_for_power_grabs = {'milestones': milestones_global}
            p_power_grabs = p_cat.p_human_power_grabs(trajectory_for_power_grabs, p_cat_params)

            # SAR to ASI duration (for human power grabs calculation)
            sar_to_asi_duration = (asi_time - sar_time) if asi_time and sar_time else None

            results['global'] = {
                'ac_time': ac_time,
                'sar_time': sar_time,
                'asi_time': asi_time,
                'handoff_duration_years': handoff_duration,
                'sar_to_asi_duration_years': sar_to_asi_duration,
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
        asi_time = milestones_covert.get('ASI', {}).get('time')

        if ac_time and sar_time:
            handoff_duration = sar_time - ac_time

            adjusted_handoff_duration = compute_research_speed_adjusted_duration(
                covert_trajectory_times, covert_ai_speedup, ac_time, sar_time,
                safety_speedup_multiplier=p_cat_params.safety_speedup_multiplier,
                max_alignment_speedup=p_cat_params.max_alignment_speedup_before_handoff
            )

            p_takeover = p_cat.p_AI_takeover(adjusted_handoff_duration, p_cat_params)

            # For human power grabs, use the trajectory with milestones
            trajectory_for_power_grabs = {'milestones': milestones_covert}
            p_power_grabs = p_cat.p_human_power_grabs(trajectory_for_power_grabs, p_cat_params)

            # SAR to ASI duration (for human power grabs calculation)
            sar_to_asi_duration = (asi_time - sar_time) if asi_time and sar_time else None

            results['covert'] = {
                'ac_time': ac_time,
                'sar_time': sar_time,
                'asi_time': asi_time,
                'handoff_duration_years': handoff_duration,
                'sar_to_asi_duration_years': sar_to_asi_duration,
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
        Dictionary containing curve data for both AI takeover and human power grabs.
        For AI takeover, duration is the research-speed-adjusted handoff duration (AC to SAR).
        For human power grabs, duration is the SAR to ASI duration.
    """
    p_cat = PCatastrophe()

    # Generate data points for both curves
    # Use log-spaced time points from ANCHOR_T1 to 20 years for log-scale x-axis
    # Start at ANCHOR_T1 since log(0) is undefined
    durations = np.logspace(np.log10(PCatastrophe.ANCHOR_T1), np.log10(20), 100)

    p_takeover = [p_cat.p_AI_takeover(d, p_cat_params) for d in durations]

    # For human power grabs, create mock trajectories with SAR-to-ASI duration
    # We use SAR at time 0 and ASI at time d for each duration d
    p_power_grabs = []
    for d in durations:
        mock_trajectory = {
            'milestones': {
                'SAR': {'time': 0.0},
                'ASI': {'time': float(d)}
            }
        }
        p_power_grabs.append(p_cat.p_human_power_grabs(mock_trajectory, p_cat_params))

    return {
        'durations': durations.tolist(),
        # Keep 'p_ai_takeover' key for frontend backward compatibility
        # (internally this is P(misalignment at handoff) - see slowdown_model.md)
        'p_ai_takeover': p_takeover,
        'p_human_power_grabs': p_power_grabs,
        # Include the anchor points for reference
        'anchor_points': {
            # Keep 'ai_takeover' key for frontend backward compatibility
            'ai_takeover': {
                't1': p_cat_params.p_misalignment_at_handoff_t1,
                't2': p_cat_params.p_misalignment_at_handoff_t2,
                't3': p_cat_params.p_misalignment_at_handoff_t3
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
    - P(human power grabs) uses SAR to ASI duration from trajectory milestones
    - P(AI takeover) uses research-speed-adjusted handoff duration
    - P(catastrophe) = 1 - (1 - P(takeover)) * (1 - P(power grabs))

    Args:
        mc_data: Monte Carlo results containing trajectory_times, speedup_percentiles, and milestones
                 This should be the proxy_project trajectory data
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
            - sar_to_asi_duration: Duration from SAR to ASI (used for human power grabs risk)
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

    # For human power grabs, use the full trajectory with milestones
    # This uses SAR-to-ASI duration from the trajectory
    trajectory_for_power_grabs = {'milestones_median': milestones}
    p_power_grabs = p_cat.p_human_power_grabs(trajectory_for_power_grabs, p_cat_params)

    # Get SAR to ASI duration for reporting
    # Note: SAR milestone uses the full name 'SAR-level-experiment-selection-skill'
    sar_milestone = milestones.get('SAR-level-experiment-selection-skill', {}) or milestones.get('SAR', {})
    sar_time = sar_milestone.get('time') if sar_milestone else None
    asi_time = milestones.get('ASI', {}).get('time')
    sar_to_asi_duration = (asi_time - sar_time) if sar_time and asi_time else None

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
            safety_speedup_multiplier=p_cat_params.safety_speedup_multiplier,
            max_alignment_speedup=p_cat_params.max_alignment_speedup_before_handoff
        )

        # Compute probabilities using the PCatastrophe class
        p_takeover = p_cat.p_AI_takeover(adjusted_handoff, p_cat_params)
        # p_power_grabs is constant (based on SAR-to-ASI duration from milestones)
        p_catastrophe = p_cat.p_domestic_takeover(trajectory_for_power_grabs, adjusted_handoff, p_cat_params)

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
        'sar_to_asi_duration': float(sar_to_asi_duration) if sar_to_asi_duration else None,
        'agreement_year': float(agreement_year)
    }


def compute_risk_reduction_over_time(
    slowdown_mc_data: Dict[str, Any],
    no_slowdown_mc_data: Dict[str, Any],
    agreement_year: float,
    p_cat_params: PCatastropheParameters
) -> Dict[str, Any]:
    """Compute risk reduction vs slowdown duration.

    Risk reduction is the difference between:
    - P(catastrophe) if there were no slowdown (using global/no-slowdown trajectory)
    - P(catastrophe) with the slowdown (using proxy_project trajectory)

    Args:
        slowdown_mc_data: Monte Carlo results for the slowdown trajectory (proxy_project)
        no_slowdown_mc_data: Monte Carlo results for the no-slowdown trajectory (global)
        agreement_year: The year when the slowdown agreement starts
        p_cat_params: PCatastropheParameters with probability anchor points

    Returns:
        Dictionary containing:
            - slowdown_duration: Duration of slowdown at each time point
            - p_catastrophe_no_slowdown: P(catastrophe) without slowdown at each duration
            - p_catastrophe_slowdown: P(catastrophe) with slowdown at each duration
            - risk_reduction: Risk reduction (no_slowdown - slowdown) at each duration
            - p_ai_takeover_reduction: AI takeover risk reduction at each duration
            - p_human_power_grabs_reduction: Human power grabs risk reduction at each duration
    """
    if not slowdown_mc_data or not no_slowdown_mc_data:
        return None

    # Get trajectory data for both scenarios
    slowdown_times = slowdown_mc_data.get('trajectory_times', [])
    slowdown_speedup = slowdown_mc_data.get('speedup_percentiles', {}).get('median', [])
    slowdown_milestones = slowdown_mc_data.get('milestones_median', {})

    no_slowdown_times = no_slowdown_mc_data.get('trajectory_times', [])
    no_slowdown_speedup = no_slowdown_mc_data.get('speedup_percentiles', {}).get('median', [])
    no_slowdown_milestones = no_slowdown_mc_data.get('milestones_median', {})

    if not slowdown_times or not slowdown_speedup or not slowdown_milestones:
        return None
    if not no_slowdown_times or not no_slowdown_speedup or not no_slowdown_milestones:
        return None

    # Get AC milestone times for both trajectories
    slowdown_ac_time = slowdown_milestones.get('AC', {}).get('time')
    no_slowdown_ac_time = no_slowdown_milestones.get('AC', {}).get('time')

    if slowdown_ac_time is None or no_slowdown_ac_time is None:
        return None

    p_cat = PCatastrophe()

    # We'll compute risk at various slowdown durations
    # Duration = time since agreement started
    # For each duration, we compute the risk if the slowdown ended at that point

    slowdown_times_arr = np.array(slowdown_times)
    no_slowdown_times_arr = np.array(no_slowdown_times)

    # Find the range of durations to compute
    # Start from when handoff begins (AC milestone) in the slowdown case
    start_time = max(agreement_year, slowdown_ac_time)

    # Get valid time indices for slowdown trajectory
    valid_indices = np.where(slowdown_times_arr >= start_time)[0]

    if len(valid_indices) == 0:
        return None

    # For human power grabs, compute once using full trajectory milestones (SAR to ASI duration)
    slowdown_trajectory_for_power_grabs = {'milestones_median': slowdown_milestones}
    no_slowdown_trajectory_for_power_grabs = {'milestones_median': no_slowdown_milestones}
    p_power_grabs_slowdown = p_cat.p_human_power_grabs(slowdown_trajectory_for_power_grabs, p_cat_params)
    p_power_grabs_no_slowdown = p_cat.p_human_power_grabs(no_slowdown_trajectory_for_power_grabs, p_cat_params)

    result_slowdown_duration = []
    result_p_catastrophe_no_slowdown = []
    result_p_catastrophe_slowdown = []
    result_risk_reduction = []
    result_p_ai_takeover_no_slowdown = []
    result_p_ai_takeover_slowdown = []
    result_p_ai_takeover_reduction = []
    result_p_human_power_grabs_no_slowdown = []
    result_p_human_power_grabs_slowdown = []
    result_p_human_power_grabs_reduction = []

    for idx in valid_indices:
        current_time = slowdown_times_arr[idx]
        slowdown_duration = current_time - agreement_year

        if slowdown_duration <= 0:
            continue

        # --- Compute P(catastrophe) WITH slowdown ---
        slowdown_handoff_duration = current_time - slowdown_ac_time
        if slowdown_handoff_duration <= 0:
            continue

        slowdown_adjusted_handoff = compute_research_speed_adjusted_duration(
            slowdown_times, slowdown_speedup, slowdown_ac_time, current_time,
            safety_speedup_multiplier=p_cat_params.safety_speedup_multiplier,
            max_alignment_speedup=p_cat_params.max_alignment_speedup_before_handoff
        )

        p_takeover_slowdown = p_cat.p_AI_takeover(slowdown_adjusted_handoff, p_cat_params)
        # p_power_grabs_slowdown is already computed above (constant, based on SAR-to-ASI)
        p_catastrophe_slowdown = p_cat.p_domestic_takeover(
            slowdown_trajectory_for_power_grabs, slowdown_adjusted_handoff, p_cat_params
        )

        # --- Compute P(catastrophe) WITHOUT slowdown ---
        # Find the corresponding time in the no-slowdown trajectory
        # The no-slowdown scenario progresses faster, so at the same calendar time,
        # more handoff time has passed
        no_slowdown_handoff_duration = current_time - no_slowdown_ac_time

        if no_slowdown_handoff_duration <= 0:
            # No-slowdown AC hasn't happened yet at this calendar time
            p_takeover_no_slowdown = 0.0
            p_catastrophe_no_slowdown = 0.0
        else:
            no_slowdown_adjusted_handoff = compute_research_speed_adjusted_duration(
                no_slowdown_times, no_slowdown_speedup, no_slowdown_ac_time, current_time,
                safety_speedup_multiplier=p_cat_params.safety_speedup_multiplier,
                max_alignment_speedup=p_cat_params.max_alignment_speedup_before_handoff
            )

            p_takeover_no_slowdown = p_cat.p_AI_takeover(no_slowdown_adjusted_handoff, p_cat_params)
            # p_power_grabs_no_slowdown is already computed above (constant, based on SAR-to-ASI)
            p_catastrophe_no_slowdown = p_cat.p_domestic_takeover(
                no_slowdown_trajectory_for_power_grabs, no_slowdown_adjusted_handoff, p_cat_params
            )

        # Risk reduction = no_slowdown risk - slowdown risk (positive means slowdown reduces risk)
        risk_reduction = p_catastrophe_no_slowdown - p_catastrophe_slowdown
        ai_takeover_reduction = p_takeover_no_slowdown - p_takeover_slowdown
        power_grabs_reduction = p_power_grabs_no_slowdown - p_power_grabs_slowdown

        result_slowdown_duration.append(float(slowdown_duration))
        result_p_catastrophe_no_slowdown.append(float(p_catastrophe_no_slowdown))
        result_p_catastrophe_slowdown.append(float(p_catastrophe_slowdown))
        result_risk_reduction.append(float(risk_reduction))
        result_p_ai_takeover_no_slowdown.append(float(p_takeover_no_slowdown))
        result_p_ai_takeover_slowdown.append(float(p_takeover_slowdown))
        result_p_ai_takeover_reduction.append(float(ai_takeover_reduction))
        result_p_human_power_grabs_no_slowdown.append(float(p_power_grabs_no_slowdown))
        result_p_human_power_grabs_slowdown.append(float(p_power_grabs_slowdown))
        result_p_human_power_grabs_reduction.append(float(power_grabs_reduction))

    if not result_slowdown_duration:
        return None

    return {
        'slowdown_duration': result_slowdown_duration,
        'p_catastrophe_no_slowdown': result_p_catastrophe_no_slowdown,
        'p_catastrophe_slowdown': result_p_catastrophe_slowdown,
        'risk_reduction': result_risk_reduction,
        'p_ai_takeover_no_slowdown': result_p_ai_takeover_no_slowdown,
        'p_ai_takeover_slowdown': result_p_ai_takeover_slowdown,
        'p_ai_takeover_reduction': result_p_ai_takeover_reduction,
        'p_human_power_grabs_no_slowdown': result_p_human_power_grabs_no_slowdown,
        'p_human_power_grabs_slowdown': result_p_human_power_grabs_slowdown,
        'p_human_power_grabs_reduction': result_p_human_power_grabs_reduction,
        'agreement_year': float(agreement_year)
    }


def compute_risk_breakdown_data(
    mc_data: Dict[str, Any],
    agreement_year: float,
    p_cat_params: PCatastropheParameters
) -> Dict[str, Any]:
    """Compute all values needed for the Risk Breakdown visualization.

    This computes all the intermediate values shown in the Risk Breakdown section,
    including:
    - P(Human Power Grabs) from SAR→ASI duration
    - P(AI Takeover) breakdown into misalignment at/after handoff
    - Pre-handoff and handoff window calculations
    - Slowdown effort adjustments
    - Alignment tax after handoff
    - Curve data for all mapping plots

    Args:
        mc_data: Monte Carlo results containing trajectory_times, speedup_percentiles, and milestones
                 This should be the proxy_project trajectory data
        agreement_year: The year when the slowdown agreement starts
        p_cat_params: PCatastropheParameters with probability anchor points and settings

    Returns:
        Dictionary containing all values for the Risk Breakdown section
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

    p_cat = PCatastrophe()
    times = np.array(trajectory_times)
    speedup = np.array(median_speedup)

    # Extract milestone times
    # Note: SAR milestone uses the full name 'SAR-level-experiment-selection-skill'
    ac_milestone = milestones.get('AC', {})
    ac_time = ac_milestone.get('time') if ac_milestone else None

    sar_milestone = milestones.get('SAR-level-experiment-selection-skill', {}) or milestones.get('SAR', {})
    sar_time = sar_milestone.get('time') if sar_milestone else None

    asi_milestone = milestones.get('ASI', {})
    asi_time = asi_milestone.get('time') if asi_milestone else None

    # Find the handoff time (when alignment speedup reaches the max cap)
    # Alignment speedup = capability_speedup × safety_multiplier
    # Handoff occurs when alignment_speedup >= max_alignment_speedup_before_handoff
    safety_multiplier = p_cat_params.safety_speedup_multiplier
    max_alignment_speedup = p_cat_params.max_alignment_speedup_before_handoff
    handoff_time = None
    for i, (t, s) in enumerate(zip(times, speedup)):
        alignment_speedup = s * safety_multiplier
        if alignment_speedup >= max_alignment_speedup:
            handoff_time = t
            break
    # Fallback to ASI time if handoff threshold never reached
    if handoff_time is None:
        handoff_time = asi_time

    present_day = p_cat_params.present_day_year

    # === Section 1: P(Human Power Grabs) ===
    sar_to_asi_duration = None
    if sar_time is not None and asi_time is not None:
        sar_to_asi_duration = asi_time - sar_time

    # Compute P(Human Power Grabs)
    trajectory_for_power_grabs = {'milestones_median': milestones}
    p_human_power_grabs = p_cat.p_human_power_grabs(trajectory_for_power_grabs, p_cat_params)

    # === Section 2: P(AI Takeover) - we'll compute components then combine ===

    # === Section 3: P(Misalignment at Handoff) breakdown ===
    safety_multiplier = p_cat_params.safety_speedup_multiplier
    relevance_discount = p_cat_params.research_relevance_of_pre_handoff_discount

    # Pre-handoff window (Present → AC)
    pre_ac_calendar_time = None
    pre_ac_avg_alignment_speedup = None
    pre_ac_adjusted_time = None

    if ac_time is not None and present_day is not None:
        pre_ac_calendar_time = max(0, ac_time - present_day)

        if pre_ac_calendar_time > 0:
            # Compute average alignment speedup during pre-AC period
            mask = (times >= present_day) & (times <= ac_time)
            if np.any(mask):
                relevant_speedup = speedup[mask]
                # Alignment speedup = capability_speedup * safety_multiplier
                alignment_speedup = relevant_speedup * safety_multiplier
                pre_ac_avg_alignment_speedup = float(np.mean(alignment_speedup))
            else:
                pre_ac_avg_alignment_speedup = 1.0

            # Adjusted time = calendar time × avg alignment speedup × relevance discount
            pre_ac_adjusted_time = pre_ac_calendar_time * pre_ac_avg_alignment_speedup * relevance_discount
        else:
            pre_ac_avg_alignment_speedup = 1.0
            pre_ac_adjusted_time = 0.0

    # Handoff window (AC → Handoff)
    handoff_window_calendar_time = None
    handoff_window_avg_alignment_speedup = None
    handoff_window_adjusted_time = None
    handoff_window_capability_time = None
    alignment_tax_during_handoff = None

    if ac_time is not None and handoff_time is not None:
        handoff_window_calendar_time = max(0, handoff_time - ac_time)

        if handoff_window_calendar_time > 0:
            # Compute average alignment speedup during handoff window
            mask = (times >= ac_time) & (times <= handoff_time)
            if np.any(mask):
                relevant_speedup = speedup[mask]
                relevant_times = times[mask]

                # Alignment speedup = capability_speedup * safety_multiplier
                alignment_speedup = relevant_speedup * safety_multiplier
                handoff_window_avg_alignment_speedup = float(np.mean(alignment_speedup))

                # Compute adjusted alignment time (integrated)
                if len(relevant_times) >= 2:
                    handoff_window_adjusted_time = float(np.trapz(alignment_speedup, relevant_times))
                    # Capability time uses full speedup (multiplier = 1.0)
                    handoff_window_capability_time = float(np.trapz(relevant_speedup, relevant_times))
                else:
                    handoff_window_adjusted_time = handoff_window_calendar_time * float(np.mean(alignment_speedup))
                    handoff_window_capability_time = handoff_window_calendar_time * float(np.mean(relevant_speedup))

                # Alignment tax during handoff = alignment time / capability time
                if handoff_window_capability_time > 0:
                    alignment_tax_during_handoff = handoff_window_adjusted_time / handoff_window_capability_time
                else:
                    alignment_tax_during_handoff = 1.0
            else:
                handoff_window_avg_alignment_speedup = 1.0
                handoff_window_adjusted_time = 0.0
                handoff_window_capability_time = 0.0
                alignment_tax_during_handoff = 1.0
        else:
            handoff_window_avg_alignment_speedup = 1.0
            handoff_window_adjusted_time = 0.0
            handoff_window_capability_time = 0.0
            alignment_tax_during_handoff = 1.0

    # === Section 4: Slowdown effort adjustment ===
    slowdown_effort_multiplier = p_cat_params.increase_in_alignment_research_effort_during_slowdown

    # Split adjusted time into before/during slowdown
    adjusted_time_before_agreement = 0.0
    adjusted_time_during_slowdown = 0.0

    if pre_ac_adjusted_time is not None:
        # Pre-AC time that occurs before agreement year
        if ac_time is not None and ac_time <= agreement_year:
            # All pre-AC time is before agreement
            adjusted_time_before_agreement += pre_ac_adjusted_time
        elif present_day is not None and present_day < agreement_year:
            # Part of pre-AC time is before agreement
            pre_agreement_calendar = min(agreement_year, ac_time or agreement_year) - present_day
            pre_agreement_fraction = pre_agreement_calendar / pre_ac_calendar_time if pre_ac_calendar_time > 0 else 0
            adjusted_time_before_agreement += pre_ac_adjusted_time * pre_agreement_fraction
            adjusted_time_during_slowdown += pre_ac_adjusted_time * (1 - pre_agreement_fraction)

    if handoff_window_adjusted_time is not None:
        if ac_time is not None:
            if ac_time >= agreement_year:
                # All handoff window is during slowdown
                adjusted_time_during_slowdown += handoff_window_adjusted_time
            elif handoff_time is not None and handoff_time > agreement_year:
                # Part of handoff window is before agreement, part during
                total_handoff_calendar = handoff_time - ac_time
                before_agreement_calendar = max(0, agreement_year - ac_time)
                during_slowdown_calendar = total_handoff_calendar - before_agreement_calendar
                if total_handoff_calendar > 0:
                    before_fraction = before_agreement_calendar / total_handoff_calendar
                    during_fraction = during_slowdown_calendar / total_handoff_calendar
                    adjusted_time_before_agreement += handoff_window_adjusted_time * before_fraction
                    adjusted_time_during_slowdown += handoff_window_adjusted_time * during_fraction
            else:
                # All handoff window is before agreement
                adjusted_time_before_agreement += handoff_window_adjusted_time

    # Total effective alignment research = before + (during × multiplier)
    total_adjusted_alignment_time = adjusted_time_before_agreement + (adjusted_time_during_slowdown * slowdown_effort_multiplier)

    # === Section 5: P(Misalignment at Handoff) ===
    p_misalignment_at_handoff = p_cat.p_misalignment_at_handoff(total_adjusted_alignment_time, p_cat_params)

    # === Section 6: P(Misalignment after Handoff) ===
    # The alignment tax is now a user-specified input: the proportion of compute spent on alignment (0 to 1)
    alignment_tax_after_handoff = p_cat_params.alignment_tax_after_handoff

    # Post-handoff calculations (for display purposes only)
    post_handoff_calendar_time = None

    if handoff_time is not None and asi_time is not None and asi_time > handoff_time:
        post_handoff_calendar_time = asi_time - handoff_time

    # P(misalignment after handoff) - uses the user-specified alignment tax directly
    p_misalignment_after_handoff = p_cat.p_misalignment_after_handoff(alignment_tax_after_handoff, p_cat_params)

    # === Combine into P(AI Takeover) ===
    p_no_misalignment_at_handoff = 1 - p_misalignment_at_handoff
    p_no_misalignment_after_handoff = 1 - (p_misalignment_after_handoff or 0)
    p_no_ai_takeover = p_no_misalignment_at_handoff * p_no_misalignment_after_handoff
    p_ai_takeover = 1 - p_no_ai_takeover

    # === Final P(Domestic Takeover) ===
    p_no_human_power_grabs = 1 - p_human_power_grabs
    p_no_domestic_takeover = p_no_ai_takeover * p_no_human_power_grabs
    p_domestic_takeover = 1 - p_no_domestic_takeover

    # === Curve data for plots ===
    # Generate curve data for the mapping plots
    durations = np.logspace(np.log10(PCatastrophe.ANCHOR_T1), np.log10(20), 100)

    # Human power grabs curve (SAR→ASI duration → P)
    # Directly use the interpolation with human power grabs anchor points
    human_power_grabs_curve = [
        p_cat._interpolate_probability(
            d,
            p_cat_params.p_human_power_grabs_t1,
            p_cat_params.p_human_power_grabs_t2,
            p_cat_params.p_human_power_grabs_t3
        )
        for d in durations
    ]

    # Misalignment at handoff curve (adjusted alignment research → P)
    misalignment_at_handoff_curve = [p_cat.p_misalignment_at_handoff(d, p_cat_params) for d in durations]

    # Alignment tax → P(Misalignment after Handoff) curve
    # The x-axis is the proportion of compute spent on alignment (0.001 to 1.0)
    alignment_tax_values = np.logspace(-3, 0, 100)  # 0.001 to 1.0
    post_handoff_misalignment_curve = [p_cat.p_misalignment_after_handoff(t, p_cat_params) for t in alignment_tax_values]

    return {
        # Top-level probabilities
        'p_domestic_takeover': float(p_domestic_takeover),
        'p_no_domestic_takeover': float(p_no_domestic_takeover),
        'p_ai_takeover': float(p_ai_takeover),
        'p_no_ai_takeover': float(p_no_ai_takeover),
        'p_human_power_grabs': float(p_human_power_grabs),
        'p_no_human_power_grabs': float(p_no_human_power_grabs),
        'p_misalignment_at_handoff': float(p_misalignment_at_handoff),
        'p_no_misalignment_at_handoff': float(p_no_misalignment_at_handoff),
        'p_misalignment_after_handoff': float(p_misalignment_after_handoff) if p_misalignment_after_handoff is not None else None,
        'p_no_misalignment_after_handoff': float(p_no_misalignment_after_handoff),

        # Section 1: Human Power Grabs
        'sar_to_asi_duration': float(sar_to_asi_duration) if sar_to_asi_duration is not None else None,

        # Section 3: Misalignment at Handoff breakdown
        'safety_speedup_multiplier': float(safety_multiplier),
        'max_alignment_speedup_before_handoff': float(p_cat_params.max_alignment_speedup_before_handoff),
        'relevance_discount': float(relevance_discount),

        # Pre-handoff window
        'pre_ac_calendar_time': float(pre_ac_calendar_time) if pre_ac_calendar_time is not None else None,
        'pre_ac_avg_alignment_speedup': float(pre_ac_avg_alignment_speedup) if pre_ac_avg_alignment_speedup is not None else None,
        'pre_ac_adjusted_time': float(pre_ac_adjusted_time) if pre_ac_adjusted_time is not None else None,

        # Handoff window
        'handoff_window_calendar_time': float(handoff_window_calendar_time) if handoff_window_calendar_time is not None else None,
        'handoff_window_avg_alignment_speedup': float(handoff_window_avg_alignment_speedup) if handoff_window_avg_alignment_speedup is not None else None,
        'handoff_window_adjusted_time': float(handoff_window_adjusted_time) if handoff_window_adjusted_time is not None else None,
        'handoff_window_capability_time': float(handoff_window_capability_time) if handoff_window_capability_time is not None else None,
        'handoff_window_relevance': 1.0,  # Always 1.0 for handoff window (research after AC has full relevance)
        'alignment_tax_during_handoff': float(alignment_tax_during_handoff) if alignment_tax_during_handoff is not None else None,

        # Section 4: Slowdown effort
        'agreement_year': float(agreement_year),
        'adjusted_time_before_agreement': float(adjusted_time_before_agreement),
        'adjusted_time_during_slowdown': float(adjusted_time_during_slowdown),
        'slowdown_effort_multiplier': float(slowdown_effort_multiplier),
        'total_adjusted_alignment_time': float(total_adjusted_alignment_time),

        # Section 6: Post-handoff
        'post_handoff_calendar_time': float(post_handoff_calendar_time) if post_handoff_calendar_time is not None else None,
        'alignment_tax_after_handoff': float(alignment_tax_after_handoff),

        # Milestone times for reference
        'ac_time': float(ac_time) if ac_time is not None else None,
        'sar_time': float(sar_time) if sar_time is not None else None,
        'asi_time': float(asi_time) if asi_time is not None else None,
        'handoff_time': float(handoff_time) if handoff_time is not None else None,

        # Anchor points for display
        'anchor_points': {
            'misalignment_at_handoff': {
                't1': p_cat_params.p_misalignment_at_handoff_t1,
                't2': p_cat_params.p_misalignment_at_handoff_t2,
                't3': p_cat_params.p_misalignment_at_handoff_t3
            },
            'human_power_grabs': {
                't1': p_cat_params.p_human_power_grabs_t1,
                't2': p_cat_params.p_human_power_grabs_t2,
                't3': p_cat_params.p_human_power_grabs_t3
            },
            'misalignment_after_handoff': {
                't1': p_cat_params.p_misalignment_after_handoff_t1,
                't2': p_cat_params.p_misalignment_after_handoff_t2,
                't3': p_cat_params.p_misalignment_after_handoff_t3
            }
        },

        # Curve data for plots
        'curves': {
            'durations': durations.tolist(),
            'human_power_grabs': human_power_grabs_curve,
            'misalignment_at_handoff': misalignment_at_handoff_curve,
            # Alignment tax → P(Misalignment after Handoff) curve
            'post_handoff': post_handoff_misalignment_curve,
            'post_handoff_x': alignment_tax_values.tolist()  # Alignment tax values (proportion of compute, 0 to 1)
        }
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
        covert_compute_data: Dict containing 'years' and 'operational_black_project'
            where operational_black_project has keys like 'p10', 'p25', 'median', 'p75', 'p90'
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
    operational = covert_compute_data.get('operational_black_project', {})

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
