"""
Clean, structured plot data extraction for the covert compute production model.

This module provides a well-organized alternative to the extract_plot_data function
in app.py, with clear separation of concerns and minimal redundancy.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from backend.classes.dark_compute_stock import H100_TPP_PER_CHIP, H100_WATTS_PER_TPP
from backend.classes.covert_fab import (
    estimate_architecture_efficiency_relative_to_h100,
    predict_watts_per_tpp_from_transistor_density,
    H100_TRANSISTOR_DENSITY_M_PER_MM2,
    PRCCovertFab
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def filter_simulations_with_fab(simulation_results):
    """Filter simulations to only include those where a covert fab was built."""
    return [
        (covert_projects, detectors)
        for covert_projects, detectors in simulation_results
        if covert_projects["prc_covert_project"].covert_fab is not None
    ]


def get_years_from_simulation(simulation_results):
    """Extract the common years array from simulation results."""
    if not simulation_results:
        return []

    covert_projects, detectors = simulation_results[0]
    us_beliefs = detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"]
    return sorted(us_beliefs.p_project_exists_update_history.keys()
                 if us_beliefs.p_project_exists_update_history else [])


def calculate_percentiles(data_array: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate 25th, 50th (median), and 75th percentiles along specified axis."""
    return (
        np.percentile(data_array, 25, axis=axis),
        np.median(data_array, axis=axis),
        np.percentile(data_array, 75, axis=axis)
    )


def convert_to_thousands(value):
    """Convert value to thousands (divide by 1000)."""
    return value / 1e3


# =============================================================================
# TIME SERIES DATA EXTRACTION
# =============================================================================

def extract_us_beliefs_over_time(simulation_results, years):
    """Extract US probability beliefs over time for each simulation."""
    us_probs_by_sim = []

    for covert_projects, detectors in simulation_results:
        us_beliefs = detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"]
        us_probs = [us_beliefs.p_covert_fab_exists for _ in years] if years else []
        us_probs_by_sim.append(us_probs)

    return us_probs_by_sim


def extract_fab_production_over_time(simulation_results, years):
    """Extract H100e production from covert fab over time for each simulation."""
    h100e_by_sim = []

    for covert_projects, detectors in simulation_results:
        covert_fab = covert_projects["prc_covert_project"].covert_fab

        if covert_fab is not None and hasattr(covert_fab, 'dark_compute_over_time'):
            h100e_counts = [covert_fab.dark_compute_over_time.get(year, 0.0) for year in years]
        else:
            h100e_counts = [0.0 for year in years]

        h100e_by_sim.append(h100e_counts)

    return h100e_by_sim


def extract_survival_rates_over_time(simulation_results, years):
    """Extract chip survival rates over time for each simulation."""
    survival_rate_by_sim = []

    for covert_projects, detectors in simulation_results:
        dark_compute_stock = covert_projects["prc_covert_project"].dark_compute_stock
        survival_rates = []

        for year in years:
            surviving_compute = dark_compute_stock.dark_compute(year)
            total_compute = dark_compute_stock.dark_compute_dead_and_alive(year)

            surviving = surviving_compute.total_h100e_tpp()
            total = total_compute.total_h100e_tpp()

            survival_rates.append(surviving / total if total > 0 else 0.0)

        survival_rate_by_sim.append(survival_rates)

    return survival_rate_by_sim


def extract_dark_compute_over_time(simulation_results, years):
    """Extract dark compute (surviving, not limited by capacity) over time."""
    dark_compute_by_sim = []

    for covert_projects, detectors in simulation_results:
        dark_compute_stock = covert_projects["prc_covert_project"].dark_compute_stock
        dark_compute = []

        for year in years:
            surviving_compute = dark_compute_stock.dark_compute(year)
            dark_compute.append(surviving_compute.total_h100e_tpp())

        dark_compute_by_sim.append(dark_compute)

    return dark_compute_by_sim


def extract_operational_dark_compute_over_time(simulation_results, years):
    """Extract operational dark compute (limited by datacenter capacity) over time."""
    operational_dark_compute_by_sim = []

    for covert_projects, detectors in simulation_results:
        operational_dark_compute = []

        for year in years:
            operational_compute = covert_projects["prc_covert_project"].operational_dark_compute(year)
            operational_dark_compute.append(operational_compute.total_h100e_tpp())

        operational_dark_compute_by_sim.append(operational_dark_compute)

    return operational_dark_compute_by_sim


def extract_datacenter_capacity_over_time(simulation_results, years, agreement_year):
    """Extract datacenter capacity in GW over time."""
    datacenter_capacity_by_sim = []

    for covert_projects, detectors in simulation_results:
        covert_datacenters = covert_projects["prc_covert_project"].covert_datacenters
        datacenter_capacity_gw = []

        for year in years:
            relative_year = year - agreement_year
            capacity_gw = covert_datacenters.get_GW_capacity(relative_year)
            datacenter_capacity_gw.append(capacity_gw)

        datacenter_capacity_by_sim.append(datacenter_capacity_gw)

    return datacenter_capacity_by_sim


def extract_datacenter_lr_over_time(simulation_results, years, agreement_year):
    """Extract likelihood ratio from datacenters over time."""
    lr_datacenters_by_sim = []

    for covert_projects, detectors in simulation_results:
        covert_datacenters = covert_projects["prc_covert_project"].covert_datacenters
        lr_datacenters_over_time = []

        for year in years:
            relative_year = year - agreement_year
            lr_datacenters = covert_datacenters.lr_from_concealed_datacenters(relative_year)
            lr_datacenters_over_time.append(lr_datacenters)

        lr_datacenters_by_sim.append(lr_datacenters_over_time)

    return lr_datacenters_by_sim


def extract_h100_years_over_time(simulation_results, years, agreement_year):
    """Extract cumulative H100-years over time for each simulation."""
    h100_years_by_sim = []

    for covert_projects, detectors in simulation_results:
        h100_years_over_time = []
        cumulative_h100_years = 0.0

        for year_idx, year in enumerate(years):
            if year_idx > 0:
                prev_year = years[year_idx - 1]
                time_increment = year - prev_year

                operational_compute = covert_projects["prc_covert_project"].operational_dark_compute(prev_year)
                h100e_at_prev_year = operational_compute.total_h100e_tpp()

                cumulative_h100_years += h100e_at_prev_year * time_increment

            h100_years_over_time.append(cumulative_h100_years)

        h100_years_by_sim.append(h100_years_over_time)

    return h100_years_by_sim


def extract_cumulative_lr_over_time(simulation_results, years):
    """Extract cumulative likelihood ratio over time for each simulation."""
    cumulative_lr_by_sim = []

    for covert_projects, detectors in simulation_results:
        cumulative_lr_over_time = []

        for year in years:
            project_lr = covert_projects["prc_covert_project"].get_cumulative_evidence_of_covert_project(year)
            cumulative_lr_over_time.append(project_lr)

        cumulative_lr_by_sim.append(cumulative_lr_over_time)

    return cumulative_lr_by_sim


# =============================================================================
# FAB-SPECIFIC DATA EXTRACTION (only for simulations with fab)
# =============================================================================

def extract_fab_lr_components_over_time(simulation_results, years):
    """Extract likelihood ratio components (inventory, procurement, other) for fab detection."""
    lr_inventory_by_sim = []
    lr_procurement_by_sim = []
    lr_other_by_sim = []

    for covert_projects, detectors in simulation_results:
        lr_inventory_over_time = []
        lr_procurement_over_time = []
        lr_other_over_time = []

        for year in years:
            lr_inventory_over_time.append(
                covert_projects["prc_covert_project"].get_fab_lr_inventory(year)
            )
            lr_procurement_over_time.append(
                covert_projects["prc_covert_project"].get_fab_lr_procurement(year)
            )
            lr_other_over_time.append(
                covert_projects["prc_covert_project"].get_fab_lr_other(year)
            )

        lr_inventory_by_sim.append(lr_inventory_over_time)
        lr_procurement_by_sim.append(lr_procurement_over_time)
        lr_other_by_sim.append(lr_other_over_time)

    return lr_inventory_by_sim, lr_procurement_by_sim, lr_other_by_sim


def extract_fab_operational_status_over_time(simulation_results, years):
    """Extract fab operational status over time."""
    is_operational_by_sim = []

    for covert_projects, detectors in simulation_results:
        is_operational = []

        for year in years:
            is_operational.append(
                covert_projects["prc_covert_project"].get_fab_is_operational(year)
            )

        is_operational_by_sim.append(is_operational)

    return is_operational_by_sim


def extract_fab_production_parameters(simulation_results, years, agreement_year):
    """Extract fab production parameters (wafer starts, chips per wafer, etc.)."""
    wafer_starts_by_sim = []
    chips_per_wafer_by_sim = []
    architecture_efficiency_by_sim = []
    compute_per_wafer_2022_arch_by_sim = []
    transistor_density_by_sim = []
    process_node_by_sim = []

    for covert_projects, detectors in simulation_results:
        wafer_starts = []
        architecture_efficiency = []
        compute_per_wafer_2022_arch = []

        for year in years:
            wafer_starts.append(
                covert_projects["prc_covert_project"].get_fab_wafer_starts_per_month()
            )

            arch_efficiency = estimate_architecture_efficiency_relative_to_h100(year, agreement_year)
            architecture_efficiency.append(arch_efficiency)

            chips_per_wafer = covert_projects["prc_covert_project"].get_fab_h100_sized_chips_per_wafer()
            transistor_density = covert_projects["prc_covert_project"].get_fab_transistor_density_relative_to_h100()
            compute_per_wafer_2022_arch.append(chips_per_wafer * transistor_density)

        wafer_starts_by_sim.append(wafer_starts)
        chips_per_wafer_by_sim.append(
            [covert_projects["prc_covert_project"].get_fab_h100_sized_chips_per_wafer()] * len(years)
        )
        architecture_efficiency_by_sim.append(architecture_efficiency)
        compute_per_wafer_2022_arch_by_sim.append(compute_per_wafer_2022_arch)
        transistor_density_by_sim.append(
            [covert_projects["prc_covert_project"].get_fab_transistor_density_relative_to_h100()] * len(years)
        )
        process_node_by_sim.append(
            covert_projects["prc_covert_project"].get_fab_process_node()
        )

    return {
        'wafer_starts': wafer_starts_by_sim,
        'chips_per_wafer': chips_per_wafer_by_sim,
        'architecture_efficiency': architecture_efficiency_by_sim,
        'compute_per_wafer_2022_arch': compute_per_wafer_2022_arch_by_sim,
        'transistor_density': transistor_density_by_sim,
        'process_node': process_node_by_sim
    }


def calculate_watts_per_tpp_by_sim(transistor_density_by_sim):
    """Calculate watts per TPP for each simulation based on transistor density."""
    watts_per_tpp_by_sim = []

    for sim_densities in transistor_density_by_sim:
        sim_watts = []
        for density_relative in sim_densities:
            density_absolute = density_relative * H100_TRANSISTOR_DENSITY_M_PER_MM2
            watts_absolute = predict_watts_per_tpp_from_transistor_density(density_absolute)
            watts_relative = watts_absolute / H100_WATTS_PER_TPP
            sim_watts.append(watts_relative)
        watts_per_tpp_by_sim.append(sim_watts)

    return watts_per_tpp_by_sim


# =============================================================================
# ENERGY BREAKDOWN
# =============================================================================

def extract_energy_breakdown_by_source(simulation_results, years):
    """Extract energy breakdown by source (initial stock vs fab production)."""
    num_years = len(years)
    num_sims = len(simulation_results)

    initial_energy_all_sims = np.zeros((num_sims, num_years))
    fab_energy_all_sims = np.zeros((num_sims, num_years))
    initial_h100e_all_sims = np.zeros((num_sims, num_years))
    fab_h100e_all_sims = np.zeros((num_sims, num_years))

    for sim_idx, (covert_projects, _) in enumerate(simulation_results):
        dark_compute_stock = covert_projects["prc_covert_project"].dark_compute_stock

        for year_idx, year in enumerate(years):
            initial_energy, fab_energy, initial_h100e, fab_h100e = \
                dark_compute_stock.dark_compute_energy_by_source(year)
            initial_energy_all_sims[sim_idx, year_idx] = initial_energy
            fab_energy_all_sims[sim_idx, year_idx] = fab_energy
            initial_h100e_all_sims[sim_idx, year_idx] = initial_h100e
            fab_h100e_all_sims[sim_idx, year_idx] = fab_h100e

    # Compute median across simulations
    initial_energy_median = np.median(initial_energy_all_sims, axis=0)
    fab_energy_median = np.median(fab_energy_all_sims, axis=0)
    initial_h100e_median = np.median(initial_h100e_all_sims, axis=0)
    fab_h100e_median = np.median(fab_h100e_all_sims, axis=0)

    # Calculate average efficiency over all years
    initial_energy_total = np.sum(initial_energy_median)
    fab_energy_total = np.sum(fab_energy_median)
    initial_h100e_total = np.sum(initial_h100e_median)
    fab_h100e_total = np.sum(fab_h100e_median)

    initial_efficiency = (initial_h100e_total / initial_energy_total) if initial_energy_total > 0 else 0
    fab_efficiency = (fab_h100e_total / fab_energy_total) if fab_energy_total > 0 else 0

    # Calculate baseline H100 efficiency
    h100_baseline_efficiency = 1e9 / (H100_TPP_PER_CHIP * H100_WATTS_PER_TPP)

    initial_efficiency_relative = initial_efficiency / h100_baseline_efficiency if h100_baseline_efficiency > 0 else 0
    fab_efficiency_relative = fab_efficiency / h100_baseline_efficiency if h100_baseline_efficiency > 0 else 0

    # Create energy array
    energy_by_source_array = np.zeros((num_years, 2))
    energy_by_source_array[:, 0] = initial_energy_median
    energy_by_source_array[:, 1] = fab_energy_median

    source_labels = [
        f"Initial Dark Compute ({initial_efficiency_relative:.2f}x energy efficiency)",
        f"Covert Fab Compute ({fab_efficiency_relative:.2f}x energy efficiency)"
    ]

    return energy_by_source_array, source_labels


# =============================================================================
# DETECTION ANALYSIS
# =============================================================================

def calculate_fab_detection_year(covert_fab, us_beliefs, years, threshold):
    """Calculate the year when fab is detected based on probability threshold."""
    if covert_fab is None:
        return None

    initial_p_fab = us_beliefs.p_covert_fab_exists if us_beliefs.p_covert_fab_exists is not None else 0.1
    prior_odds_fab = initial_p_fab / (1 - initial_p_fab) if initial_p_fab < 1.0 else 1e10

    for year in years:
        if hasattr(covert_fab, 'detection_updates') and year in covert_fab.detection_updates:
            fab_lr = covert_fab.detection_updates[year]
            posterior_odds_fab = prior_odds_fab * fab_lr
            p_fab_detected = posterior_odds_fab / (1 + posterior_odds_fab)

            if p_fab_detected >= threshold:
                return year

    return None


def extract_fab_compute_at_detection(simulation_results, years, threshold, total_simulations_with_fab):
    """Extract compute produced by fab at detection time."""
    compute_at_detection = []
    operational_time_at_detection = []

    for covert_projects, detectors in simulation_results:
        covert_fab = covert_projects["prc_covert_project"].covert_fab

        if covert_fab is None:
            continue

        us_beliefs = detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"]
        h100e_over_time = covert_fab.dark_compute_over_time if hasattr(covert_fab, 'dark_compute_over_time') else {}

        sim_years = sorted(us_beliefs.p_project_exists_update_history.keys()
                          if us_beliefs.p_project_exists_update_history else [])

        detection_year = calculate_fab_detection_year(covert_fab, us_beliefs, sim_years, threshold)

        if detection_year is not None:
            available_years = [y for y in h100e_over_time.keys() if y <= detection_year]
            h100e_at_detection = h100e_over_time[max(available_years)] if available_years else 0.0
        else:
            # Never detected - use final year
            final_year = years[-1]
            available_years = [y for y in h100e_over_time.keys() if y <= final_year]
            h100e_at_detection = h100e_over_time[max(available_years)] if available_years else 0.0
            detection_year = final_year

        compute_at_detection.append(h100e_at_detection)

        # Calculate operational time
        construction_start = covert_fab.construction_start_year
        construction_duration = covert_fab.construction_duration
        operational_start = construction_start + construction_duration
        operational_time = max(0.0, detection_year - operational_start)
        operational_time_at_detection.append(operational_time)

    return compute_at_detection, operational_time_at_detection


def calculate_ccdf(values, total_count):
    """Calculate complementary cumulative distribution function (CCDF)."""
    if not values:
        return []

    values_sorted = np.sort(values)
    unique_x = []
    ccdf_y = []
    seen_values = set()

    for i, x in enumerate(values_sorted):
        if x not in seen_values:
            seen_values.add(x)
            # Find last occurrence of this x value
            last_idx = i
            while last_idx + 1 < len(values_sorted) and values_sorted[last_idx + 1] == x:
                last_idx += 1
            # CCDF at x = (number of values > x) / total
            num_greater = total_count - (last_idx + 1)
            ccdf = num_greater / total_count
            unique_x.append(float(x))
            ccdf_y.append(float(ccdf))

    return [{"x": x, "y": y} for x, y in zip(unique_x, ccdf_y)]


def extract_fab_detection_statistics(simulation_results, years, detection_thresholds, likelihood_ratios):
    """Extract comprehensive fab detection statistics for multiple thresholds."""
    total_simulations_with_fab = sum(
        1 for cp, _ in simulation_results
        if cp["prc_covert_project"].covert_fab is not None
    )

    compute_ccdfs = {}
    op_time_ccdfs = {}

    for threshold in detection_thresholds:
        compute_at_detection, operational_time_at_detection = extract_fab_compute_at_detection(
            simulation_results, years, threshold, total_simulations_with_fab
        )

        compute_ccdfs[threshold] = calculate_ccdf(compute_at_detection, total_simulations_with_fab)
        op_time_ccdfs[threshold] = calculate_ccdf(operational_time_at_detection, total_simulations_with_fab)

    return compute_ccdfs, op_time_ccdfs


def extract_individual_fab_detection_data(simulation_results, years, threshold, app_params):
    """Extract individual fab detection data for dashboard display."""
    individual_h100e = []
    individual_time = []
    individual_process_nodes = []
    individual_energy = []

    for covert_projects, detectors in simulation_results:
        covert_fab = covert_projects["prc_covert_project"].covert_fab

        if covert_fab is None:
            continue

        us_beliefs = detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"]
        h100e_over_time = covert_fab.dark_compute_over_time if hasattr(covert_fab, 'dark_compute_over_time') else {}

        sim_years = sorted(us_beliefs.p_project_exists_update_history.keys()
                          if us_beliefs.p_project_exists_update_history else [])

        detection_year = calculate_fab_detection_year(covert_fab, us_beliefs, sim_years, threshold)

        if detection_year is not None:
            available_years = [y for y in h100e_over_time.keys() if y <= detection_year]
            h100e_at_detection = h100e_over_time[max(available_years)] if available_years else 0.0
        else:
            final_year = years[-1]
            available_years = [y for y in h100e_over_time.keys() if y <= final_year]
            h100e_at_detection = h100e_over_time[max(available_years)] if available_years else 0.0
            detection_year = final_year

        construction_start = covert_fab.construction_start_year
        construction_duration = covert_fab.construction_duration
        operational_start = construction_start + construction_duration
        operational_time = max(0.0, detection_year - operational_start)

        individual_h100e.append(h100e_at_detection)
        individual_time.append(operational_time)
        individual_process_nodes.append(covert_fab.process_node.value)

        energy_gw = (h100e_at_detection * H100_TPP_PER_CHIP * H100_WATTS_PER_TPP /
                    app_params.covert_project_parameters.initial_compute_stock_parameters.energy_efficiency_relative_to_h100 / 1e9)
        individual_energy.append(energy_gw)

    return individual_h100e, individual_time, individual_process_nodes, individual_energy


def calculate_project_detection_year(us_beliefs, years, likelihood_ratio, p_project_exists):
    """Calculate detection year for covert project based on likelihood ratio threshold."""
    if not years:
        return None

    initial_p_project = us_beliefs.p_project_exists_update_history[min(years)][-1]['current_p_project_exists'] \
        if years else (us_beliefs.p_project_exists if us_beliefs.p_project_exists is not None else 0.2)

    prior_odds_project = initial_p_project / (1 - initial_p_project) if initial_p_project < 1.0 else 1e10

    for year in years:
        p_project = us_beliefs.p_project_exists_update_history[year][-1]['current_p_project_exists']
        current_odds = p_project / (1 - p_project) if p_project < 1.0 else 1e10
        odds_ratio = current_odds / prior_odds_project

        if odds_ratio >= likelihood_ratio:
            return year

    return None


def calculate_h100_years_until_detection(covert_projects, years, agreement_year, detection_year):
    """Calculate cumulative H100-years until detection (or end of simulation)."""
    end_year = detection_year if detection_year is not None else max(years)
    h100_years = 0.0
    years_to_integrate = [y for y in years if agreement_year <= y <= end_year]

    for i in range(len(years_to_integrate) - 1):
        year = years_to_integrate[i]
        next_year = years_to_integrate[i + 1]
        time_increment = next_year - year

        operational_at_year = covert_projects["prc_covert_project"].operational_dark_compute(year)
        h100e_at_year = operational_at_year.total_h100e_tpp()

        h100_years += h100e_at_year * time_increment

    return h100_years


def extract_project_h100_years_ccdfs(simulation_results, years, agreement_year, likelihood_ratios, p_project_exists):
    """Calculate CCDFs for H100-years at different detection thresholds for the covert project."""
    project_h100_years_ccdfs = {}
    total_simulations = len(simulation_results)

    for lr in likelihood_ratios:
        h100_years_at_threshold = []

        for covert_projects, detectors in simulation_results:
            us_beliefs = detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"]
            sim_years = sorted(us_beliefs.p_project_exists_update_history.keys()
                             if us_beliefs.p_project_exists_update_history else [])

            detection_year = calculate_project_detection_year(us_beliefs, sim_years, lr, p_project_exists)
            h100_years = calculate_h100_years_until_detection(covert_projects, sim_years, agreement_year, detection_year)
            h100_years_at_threshold.append(h100_years)

        project_h100_years_ccdfs[lr] = calculate_ccdf(h100_years_at_threshold, total_simulations)

    return project_h100_years_ccdfs


def extract_individual_project_detection_data(simulation_results, years, agreement_year, threshold, app_params):
    """Extract individual project detection data for dashboard display."""
    individual_h100e = []
    individual_energy = []
    individual_time = []
    individual_h100_years = []

    for covert_projects, detectors in simulation_results:
        us_beliefs = detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"]
        sim_years = sorted(us_beliefs.p_project_exists_update_history.keys()
                          if us_beliefs.p_project_exists_update_history else [])

        detection_year = calculate_project_detection_year(us_beliefs, sim_years, threshold,
                                                          app_params.covert_project_parameters.p_project_exists)

        if detection_year is not None:
            operational_compute = covert_projects["prc_covert_project"].operational_dark_compute(detection_year)
        else:
            final_year = max(sim_years)
            operational_compute = covert_projects["prc_covert_project"].operational_dark_compute(final_year)
            detection_year = final_year

        operational_h100e = operational_compute.total_h100e_tpp()

        energy_gw = (operational_h100e * H100_TPP_PER_CHIP * H100_WATTS_PER_TPP /
                    app_params.covert_project_parameters.initial_compute_stock_parameters.energy_efficiency_relative_to_h100 / 1e9)

        time_operational = detection_year - agreement_year
        h100_years = calculate_h100_years_until_detection(covert_projects, sim_years, agreement_year, detection_year)

        individual_h100e.append(operational_h100e)
        individual_energy.append(energy_gw)
        individual_time.append(time_operational)
        individual_h100_years.append(h100_years)

    return individual_h100e, individual_energy, individual_time, individual_h100_years


# =============================================================================
# MAIN EXTRACTION FUNCTION
# =============================================================================

def extract_plot_data(model, p_project_exists, app_params, likelihood_ratios):
    """
    Extract all plot data from model simulation results.

    Args:
        model: Model instance with simulation results
        p_project_exists: Prior probability that covert project exists
        app_params: Application parameters
        likelihood_ratios: List of likelihood ratio thresholds for detection plots

    Returns:
        Dictionary containing all plot data
    """
    if not model.simulation_results:
        return {"error": "No simulation results"}

    agreement_year = model.year_us_prc_agreement_goes_into_force
    years = get_years_from_simulation(model.simulation_results)

    # Track which simulations have fabs built
    fab_built_in_sim = [
        cp["prc_covert_project"].covert_fab is not None
        for cp, _ in model.simulation_results
    ]

    # Extract time series data (all simulations)
    us_probs_by_sim = extract_us_beliefs_over_time(model.simulation_results, years)
    h100e_by_sim = extract_fab_production_over_time(model.simulation_results, years)
    survival_rate_by_sim = extract_survival_rates_over_time(model.simulation_results, years)
    dark_compute_by_sim = extract_dark_compute_over_time(model.simulation_results, years)
    operational_dark_compute_by_sim = extract_operational_dark_compute_over_time(model.simulation_results, years)
    datacenter_capacity_by_sim = extract_datacenter_capacity_over_time(model.simulation_results, years, agreement_year)
    lr_datacenters_by_sim = extract_datacenter_lr_over_time(model.simulation_results, years, agreement_year)
    h100_years_by_sim = extract_h100_years_over_time(model.simulation_results, years, agreement_year)
    cumulative_lr_by_sim = extract_cumulative_lr_over_time(model.simulation_results, years)

    # Extract fab-specific data (only simulations with fab)
    simulations_with_fab = filter_simulations_with_fab(model.simulation_results)

    lr_inventory_by_sim, lr_procurement_by_sim, lr_other_by_sim = \
        extract_fab_lr_components_over_time(model.simulation_results, years)

    is_operational_by_sim = extract_fab_operational_status_over_time(model.simulation_results, years)

    fab_params = extract_fab_production_parameters(model.simulation_results, years, agreement_year)

    watts_per_tpp_by_sim = calculate_watts_per_tpp_by_sim(fab_params['transistor_density'])

    # Energy breakdown
    energy_by_source_array, source_labels = extract_energy_breakdown_by_source(model.simulation_results, years)

    # Filter arrays to only include simulations where fab was built
    us_probs_with_fab = [sim for i, sim in enumerate(us_probs_by_sim) if fab_built_in_sim[i]]
    h100e_with_fab = [sim for i, sim in enumerate(h100e_by_sim) if fab_built_in_sim[i]]

    lr_inventory_with_fab = [sim for i, sim in enumerate(lr_inventory_by_sim) if fab_built_in_sim[i]]
    lr_procurement_with_fab = [sim for i, sim in enumerate(lr_procurement_by_sim) if fab_built_in_sim[i]]
    lr_other_with_fab = [sim for i, sim in enumerate(lr_other_by_sim) if fab_built_in_sim[i]]

    is_operational_with_fab = [sim for i, sim in enumerate(is_operational_by_sim) if fab_built_in_sim[i]]
    wafer_starts_with_fab = [sim for i, sim in enumerate(fab_params['wafer_starts']) if fab_built_in_sim[i]]
    chips_per_wafer_with_fab = [sim for i, sim in enumerate(fab_params['chips_per_wafer']) if fab_built_in_sim[i]]
    architecture_efficiency_with_fab = [sim for i, sim in enumerate(fab_params['architecture_efficiency']) if fab_built_in_sim[i]]
    compute_per_wafer_2022_arch_with_fab = [sim for i, sim in enumerate(fab_params['compute_per_wafer_2022_arch']) if fab_built_in_sim[i]]
    transistor_density_with_fab = [sim for i, sim in enumerate(fab_params['transistor_density']) if fab_built_in_sim[i]]
    watts_per_tpp_with_fab = [sim for i, sim in enumerate(watts_per_tpp_by_sim) if fab_built_in_sim[i]]
    process_node_with_fab = [sim for i, sim in enumerate(fab_params['process_node']) if fab_built_in_sim[i]]

    # Calculate statistics using filtered data where appropriate
    years_array = np.array(years)

    # Calculate percentiles for all time series
    us_probs_percentiles = calculate_percentiles(np.array(us_probs_with_fab) if len(us_probs_with_fab) > 0 else np.array(us_probs_by_sim))
    h100e_percentiles = calculate_percentiles(np.array(h100e_with_fab) if len(h100e_with_fab) > 0 else np.array(h100e_by_sim))
    survival_rate_percentiles = calculate_percentiles(np.array(survival_rate_by_sim))
    dark_compute_percentiles = calculate_percentiles(np.array(dark_compute_by_sim))
    operational_dark_compute_percentiles = calculate_percentiles(np.array(operational_dark_compute_by_sim))
    datacenter_capacity_percentiles = calculate_percentiles(np.array(datacenter_capacity_by_sim))
    lr_datacenters_array = np.array(lr_datacenters_by_sim)
    lr_datacenters_percentiles = calculate_percentiles(lr_datacenters_array)
    h100_years_percentiles = calculate_percentiles(np.array(h100_years_by_sim))
    cumulative_lr_percentiles = calculate_percentiles(np.array(cumulative_lr_by_sim))

    # Detection statistics
    detection_thresholds = [0.5]
    dashboard_threshold = detection_thresholds[-1]

    total_simulations_with_fab = sum(fab_built_in_sim)

    compute_ccdfs, op_time_ccdfs = extract_fab_detection_statistics(
        model.simulation_results, years, detection_thresholds, likelihood_ratios
    )

    individual_h100e, individual_time, individual_process_nodes, individual_energy = \
        extract_individual_fab_detection_data(model.simulation_results, years, dashboard_threshold, app_params)

    project_h100_years_ccdfs = extract_project_h100_years_ccdfs(
        model.simulation_results, years, agreement_year, likelihood_ratios, p_project_exists
    )

    individual_project_h100e, individual_project_energy, individual_project_time, individual_project_h100_years = \
        extract_individual_project_detection_data(model.simulation_results, years, agreement_year,
                                                 dashboard_threshold, app_params)

    # Calculate detection probability for datacenters
    detection_threshold = 5.0
    datacenter_detection_prob = np.mean(lr_datacenters_array >= detection_threshold, axis=0)

    # LR components for fab detection (only for simulations with fab)
    lr_inventory_array = np.array(lr_inventory_with_fab) if len(lr_inventory_with_fab) > 0 else np.array(lr_inventory_by_sim)
    lr_procurement_array = np.array(lr_procurement_with_fab) if len(lr_procurement_with_fab) > 0 else np.array(lr_procurement_by_sim)
    lr_other_array = np.array(lr_other_with_fab) if len(lr_other_with_fab) > 0 else np.array(lr_other_by_sim)

    lr_components = {
        "inventory_median": np.median(lr_inventory_array, axis=0).tolist(),
        "inventory_individual": [list(sim) for sim in lr_inventory_array],
        "procurement_median": np.median(lr_procurement_array, axis=0).tolist(),
        "procurement_individual": [list(sim) for sim in lr_procurement_array],
        "other_median": np.median(lr_other_array, axis=0).tolist(),
        "other_individual": [list(sim) for sim in lr_other_array]
    }

    # Compute factors (only for simulations with fab)
    is_operational_array = np.array(is_operational_with_fab) if len(is_operational_with_fab) > 0 else np.array(is_operational_by_sim)
    wafer_starts_array = np.array(wafer_starts_with_fab) if len(wafer_starts_with_fab) > 0 else np.array(fab_params['wafer_starts'])
    chips_per_wafer_array = np.array(chips_per_wafer_with_fab) if len(chips_per_wafer_with_fab) > 0 else np.array(fab_params['chips_per_wafer'])
    architecture_efficiency_array = np.array(architecture_efficiency_with_fab) if len(architecture_efficiency_with_fab) > 0 else np.array(fab_params['architecture_efficiency'])
    compute_per_wafer_2022_arch_array = np.array(compute_per_wafer_2022_arch_with_fab) if len(compute_per_wafer_2022_arch_with_fab) > 0 else np.array(fab_params['compute_per_wafer_2022_arch'])
    transistor_density_array = np.array(transistor_density_with_fab) if len(transistor_density_with_fab) > 0 else np.array(fab_params['transistor_density'])
    watts_per_tpp_array = np.array(watts_per_tpp_with_fab) if len(watts_per_tpp_with_fab) > 0 else np.array(watts_per_tpp_by_sim)

    compute_factors = {
        "is_operational_median": np.mean(is_operational_array, axis=0).tolist(),
        "is_operational_individual": [list(sim) for sim in is_operational_array],
        "wafer_starts_median": np.median(wafer_starts_array, axis=0).tolist(),
        "wafer_starts_individual": [list(sim) for sim in wafer_starts_array],
        "chips_per_wafer_median": np.median(chips_per_wafer_array, axis=0).tolist(),
        "chips_per_wafer_individual": [list(sim) for sim in chips_per_wafer_array],
        "architecture_efficiency_median": np.median(architecture_efficiency_array, axis=0).tolist(),
        "architecture_efficiency_individual": [list(sim) for sim in architecture_efficiency_array],
        "compute_per_wafer_2022_arch_median": np.median(compute_per_wafer_2022_arch_array, axis=0).tolist(),
        "compute_per_wafer_2022_arch_individual": [list(sim) for sim in compute_per_wafer_2022_arch_array],
        "transistor_density_median": np.median(transistor_density_array, axis=0).tolist(),
        "transistor_density_individual": [list(sim) for sim in transistor_density_array],
        "watts_per_tpp_median": np.median(watts_per_tpp_array, axis=0).tolist(),
        "watts_per_tpp_individual": [list(sim) for sim in watts_per_tpp_array],
        "process_node_by_sim": process_node_with_fab if len(process_node_with_fab) > 0 else fab_params['process_node']
    }

    # Calculate architecture efficiency at agreement year
    architecture_efficiency_at_agreement = estimate_architecture_efficiency_relative_to_h100(agreement_year, agreement_year)

    # Helper to format percentiles for output (p25, median, p75)
    def format_percentiles(percentiles, scale_fn=None):
        p25, median, p75 = percentiles
        if scale_fn:
            return {
                "p25": scale_fn(p25).tolist(),
                "median": scale_fn(median).tolist(),
                "p75": scale_fn(p75).tolist()
            }
        return {
            "p25": p25.tolist(),
            "median": median.tolist(),
            "p75": p75.tolist()
        }

    # Return results structured by page sections
    return {
        # =====================================================================
        # DARK COMPUTE MODEL
        # =====================================================================
        "dark_compute_model": {
            "years": years_array.tolist(),

            # -------------------------------------------------------------------
            # Dashboard values (80th percentile calculations)
            # -------------------------------------------------------------------
            "project_80th_h100_years": individual_project_h100_years,
            "project_80th_h100e": individual_project_h100e,
            "project_80th_time": individual_project_time,

            # -------------------------------------------------------------------
            # H100-Years CCDF Plot
            # -------------------------------------------------------------------
            "h100_years_ccdf": project_h100_years_ccdfs,

            # -------------------------------------------------------------------
            # H100-Years and Evidence Over Time Plot
            # -------------------------------------------------------------------
            "h100_years": format_percentiles(h100_years_percentiles),
            "cumulative_lr": format_percentiles(cumulative_lr_percentiles),

            # -------------------------------------------------------------------
            # Dark Compute Stock Breakdown
            # -------------------------------------------------------------------
            "initial_dark_compute": format_percentiles(dark_compute_percentiles, convert_to_thousands),  # Initial stock portion
            "covert_fab_flow": format_percentiles(h100e_percentiles, convert_to_thousands),  # Fab production flow
            "survival_rate": format_percentiles(survival_rate_percentiles),  # Chip survival rate
            "total_dark_compute": format_percentiles(dark_compute_percentiles, convert_to_thousands),  # Combined result

            # -------------------------------------------------------------------
            # Data Center Capacity Breakdown
            # -------------------------------------------------------------------
            "datacenter_capacity": format_percentiles(datacenter_capacity_percentiles),
            "dark_compute_energy": energy_by_source_array.tolist(),
            "energy_source_labels": source_labels,
            "operational_dark_compute": format_percentiles(operational_dark_compute_percentiles, convert_to_thousands),

            # -------------------------------------------------------------------
            # Strength of Evidence Breakdown
            # -------------------------------------------------------------------
            # TODO: Add LR components for initial stock, diverted SME, other intel
            "posterior_prob_project": format_percentiles(us_probs_percentiles),
        },

        # Legacy time_series for backward compatibility (can be removed later)
        "time_series": {
            "years": years_array.tolist(),
            "us_prob_median": us_probs_percentiles[1].tolist(),
            "us_prob_p25": us_probs_percentiles[0].tolist(),
            "us_prob_p75": us_probs_percentiles[2].tolist(),
            "h100e_median": convert_to_thousands(h100e_percentiles[1]).tolist(),
            "h100e_p25": convert_to_thousands(h100e_percentiles[0]).tolist(),
            "h100e_p75": convert_to_thousands(h100e_percentiles[2]).tolist(),
            "h100_years_median": h100_years_percentiles[1].tolist(),
            "h100_years_p25": h100_years_percentiles[0].tolist(),
            "h100_years_p75": h100_years_percentiles[2].tolist(),
            "cumulative_lr_median": cumulative_lr_percentiles[1].tolist(),
            "cumulative_lr_p25": cumulative_lr_percentiles[0].tolist(),
            "cumulative_lr_p75": cumulative_lr_percentiles[2].tolist(),
            "survival_rate_median": survival_rate_percentiles[1].tolist(),
            "survival_rate_p25": survival_rate_percentiles[0].tolist(),
            "survival_rate_p75": survival_rate_percentiles[2].tolist(),
            "dark_compute_median": convert_to_thousands(dark_compute_percentiles[1]).tolist(),
            "dark_compute_p25": convert_to_thousands(dark_compute_percentiles[0]).tolist(),
            "dark_compute_p75": convert_to_thousands(dark_compute_percentiles[2]).tolist(),
            "operational_dark_compute_median": convert_to_thousands(operational_dark_compute_percentiles[1]).tolist(),
            "operational_dark_compute_p25": convert_to_thousands(operational_dark_compute_percentiles[0]).tolist(),
            "operational_dark_compute_p75": convert_to_thousands(operational_dark_compute_percentiles[2]).tolist(),
            "datacenter_capacity_median": datacenter_capacity_percentiles[1].tolist(),
            "datacenter_capacity_p25": datacenter_capacity_percentiles[0].tolist(),
            "datacenter_capacity_p75": datacenter_capacity_percentiles[2].tolist(),
            "lr_datacenters_median": lr_datacenters_percentiles[1].tolist(),
            "lr_datacenters_p25": lr_datacenters_percentiles[0].tolist(),
            "lr_datacenters_p75": lr_datacenters_percentiles[2].tolist(),
            "datacenter_detection_prob": datacenter_detection_prob.tolist(),
            "energy_by_source": energy_by_source_array.tolist(),
            "source_labels": source_labels,
            "individual_us_probs": [list(sim) for sim in us_probs_by_sim],
            "individual_h100e": [list(convert_to_thousands(np.array(sim))) for sim in h100e_by_sim],
            "individual_us_probs_with_fab": [list(sim) for sim in us_probs_with_fab],
            "individual_h100e_with_fab": [list(convert_to_thousands(np.array(sim))) for sim in h100e_with_fab],
            "fab_built": fab_built_in_sim
        },
        "compute_ccdf": compute_ccdfs.get(dashboard_threshold, []),
        "compute_ccdfs": compute_ccdfs,
        "op_time_ccdf": op_time_ccdfs.get(dashboard_threshold, []),
        "op_time_ccdfs": op_time_ccdfs,
        "likelihood_ratios": likelihood_ratios,
        "lr_components": lr_components,
        "compute_factors": compute_factors,
        "architecture_efficiency_at_agreement": architecture_efficiency_at_agreement,
        "num_simulations": len(model.simulation_results),
        "prob_fab_built": sum(fab_built_in_sim) / len(fab_built_in_sim) if len(fab_built_in_sim) > 0 else 0.0,
        "individual_h100e_before_detection": individual_h100e,
        "individual_time_before_detection": individual_time,
        "individual_process_node": individual_process_nodes,
        "individual_energy_before_detection": individual_energy,
        "watts_per_tpp_curve": PRCCovertFab.watts_per_tpp_relative_to_H100(),
        "individual_project_h100e_before_detection": individual_project_h100e,
        "individual_project_energy_before_detection": individual_project_energy,
        "individual_project_time_before_detection": individual_project_time,
        "individual_project_h100_years_before_detection": individual_project_h100_years,
        "project_h100_years_ccdfs": project_h100_years_ccdfs,
        "p_project_exists": p_project_exists
    }
