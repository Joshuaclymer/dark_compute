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

# Configure likelihood ratio thresholds for detection plots
# Update this list to change all plots automatically
LIKELIHOOD_RATIOS = [1, 3, 5]


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


def filter_to_with_fab(data_by_sim, fab_built_mask):
    """Filter simulation data to only include simulations where a fab was built."""
    return [sim for i, sim in enumerate(data_by_sim) if fab_built_mask[i]]


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

def extract_project_beliefs_over_time(simulation_results, years):
    """Extract US beliefs about covert project existence over time."""
    project_beliefs_by_sim = []

    for covert_projects, detectors in simulation_results:
        us_beliefs = detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"]

        # Extract the final belief for each year from update history
        beliefs_over_time = []
        for year in years:
            if year in us_beliefs.p_project_exists_update_history:
                # Get the last update for this year
                updates = us_beliefs.p_project_exists_update_history[year]
                if updates:
                    # Use the final posterior probability for this year
                    beliefs_over_time.append(updates[-1]['current_p_project_exists'])
                else:
                    beliefs_over_time.append(us_beliefs.p_project_exists)
            else:
                beliefs_over_time.append(us_beliefs.p_project_exists)

        project_beliefs_by_sim.append(beliefs_over_time)

    return project_beliefs_by_sim


def extract_fab_production_over_time(simulation_results, years):
    """Extract H100e production from covert fab over time for each simulation."""
    h100e_by_sim = []

    for covert_projects, detectors in simulation_results:
        covert_fab = covert_projects["prc_covert_project"].covert_fab

        if covert_fab is not None:
            # Get cumulative production for all recorded years
            cumulative_by_year = covert_fab.get_cumulative_compute_production_over_time()
            # Map to the requested years array
            h100e_counts = [cumulative_by_year.get(year, 0.0) for year in years]
        else:
            h100e_counts = [0.0 for _ in years]

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


def extract_project_lr_components_over_time(simulation_results, years):
    """Extract likelihood ratio components for covert project detection (initial, SME, other intel)."""
    lr_initial_by_sim = []
    lr_sme_by_sim = []
    lr_other_by_sim = []

    for covert_projects, detectors in simulation_results:
        # Initial stock LR is constant over time
        lr_initial = covert_projects["prc_covert_project"].get_lr_initial()
        lr_initial_over_time = [lr_initial for _ in years]

        # SME LR (from fab procurement/inventory) is constant over time
        lr_sme = covert_projects["prc_covert_project"].get_lr_sme()
        lr_sme_over_time = [lr_sme for _ in years]

        # Other intelligence (workers, etc.) varies over time
        lr_other_over_time = []
        for year in years:
            lr_other_over_time.append(
                covert_projects["prc_covert_project"].get_lr_other(year)
            )

        lr_initial_by_sim.append(lr_initial_over_time)
        lr_sme_by_sim.append(lr_sme_over_time)
        lr_other_by_sim.append(lr_other_over_time)

    return lr_initial_by_sim, lr_sme_by_sim, lr_other_by_sim


def extract_fab_combined_lr_over_time(simulation_results, years):
    """Extract cumulative combined likelihood ratio from fab's cumulative_detection_likelihood_ratio method."""
    combined_lr_by_sim = []

    for covert_projects, _ in simulation_results:
        covert_fab = covert_projects["prc_covert_project"].covert_fab

        if covert_fab is not None:
            # Call the fab's cumulative_detection_likelihood_ratio method for each year
            combined_lr_over_time = [
                covert_fab.cumulative_detection_likelihood_ratio(year)
                for year in years
            ]
        else:
            # If no fab, LR is 1.0 (no evidence)
            combined_lr_over_time = [1.0 for _ in years]

        combined_lr_by_sim.append(combined_lr_over_time)

    return combined_lr_by_sim


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

    # DEBUG: Log lr_other values
    if lr_other_by_sim:
        lr_other_array = np.array(lr_other_by_sim)

        # Plot single simulation to visualize
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(years, lr_other_by_sim[0], 'b-', linewidth=2, label='Simulation 1')
        if len(lr_other_by_sim) > 1:
            plt.plot(years, lr_other_by_sim[1], 'r-', linewidth=2, label='Simulation 2', alpha=0.7)
        plt.xlabel('Year')
        plt.ylabel('LR (other intelligence)')
        plt.title('Evidence from Other Intelligence Sources - Single Simulation Time Series')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.savefig('/tmp/lr_other_debug.png', dpi=150, bbox_inches='tight')
        plt.close()

    return lr_inventory_by_sim, lr_procurement_by_sim, lr_other_by_sim


def extract_fab_operational_status_over_time(simulation_results, years):
    """Extract fab operational status over time.

    Returns the proportion of simulations (that built fabs) where construction
    is complete at each time point.
    """
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
# INITIAL DARK COMPUTE STOCK ANALYSIS
# =============================================================================

def extract_initial_stock_data(simulation_results, likelihood_ratios, diversion_proportion):
    """Extract initial dark compute stock data from simulation results."""
    initial_prc_stock_samples = []
    initial_compute_samples = []
    lr_prc_accounting_samples = []
    lr_global_accounting_samples = []
    lr_combined_samples = []

    for covert_projects, detectors in simulation_results:
        dark_compute_stock = covert_projects["prc_covert_project"].dark_compute_stock

        # Extract initial stock values
        initial_prc_stock_samples.append(dark_compute_stock.initial_prc_stock)
        initial_compute_samples.append(dark_compute_stock.initial_prc_dark_compute)

        # Extract likelihood ratios
        lr_prc = dark_compute_stock.lr_from_prc_compute_accounting
        lr_global = dark_compute_stock.lr_from_global_compute_production_accounting
        lr_prc_accounting_samples.append(lr_prc)
        lr_global_accounting_samples.append(lr_global)

        combined_lr = lr_prc * lr_global
        lr_combined_samples.append(combined_lr)

    # Calculate detection probabilities for each threshold
    detection_probs = {}
    for threshold in likelihood_ratios:
        num_detected = sum(1 for lr in lr_combined_samples if lr >= threshold)
        detection_probs[f"{threshold}x"] = num_detected / len(lr_combined_samples)

    return {
        'initial_prc_stock_samples': initial_prc_stock_samples,
        'initial_compute_stock_samples': initial_compute_samples,
        'diversion_proportion': diversion_proportion,
        'lr_prc_accounting_samples': lr_prc_accounting_samples,
        'lr_global_accounting_samples': lr_global_accounting_samples,
        'lr_combined_samples': lr_combined_samples,
        'initial_dark_compute_detection_probs': detection_probs
    }


# =============================================================================
# DETECTION ANALYSIS
# =============================================================================

def calculate_fab_detection_year(covert_fab, years, threshold):
    """Calculate the year when fab is detected based on likelihood ratio threshold.

    Uses the fab's cumulative_detection_likelihood_ratio method which combines:
    - inventory accounting (constant)
    - procurement accounting (constant)
    - other intelligence (grows over time with workers)

    Detection occurs the first year the cumulative LR exceeds the threshold.
    """
    if covert_fab is None:
        return None

    for year in years:
        # Get the fab's cumulative LR (inventory × procurement × other) for this year
        year_lr = covert_fab.cumulative_detection_likelihood_ratio(year)

        # Check if LR exceeds threshold
        if year_lr >= threshold:
            return year

    return None


def extract_fab_compute_at_detection(simulation_results, years, threshold):
    """Extract compute produced by fab at detection time.

    Only includes simulations where a covert fab was built.
    """
    compute_at_detection = []
    operational_time_at_detection = []

    for covert_projects, detectors in simulation_results:
        covert_fab = covert_projects["prc_covert_project"].covert_fab

        if covert_fab is None:
            continue

        # Get cumulative compute production from the fab over time
        h100e_over_time = covert_fab.get_cumulative_compute_production_over_time()

        # Use simulation years to calculate detection
        us_beliefs = detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"]
        sim_years = sorted(us_beliefs.p_project_exists_update_history.keys()
                          if us_beliefs.p_project_exists_update_history else [])

        detection_year = calculate_fab_detection_year(covert_fab, sim_years, threshold)

        if detection_year is not None:
            # Get compute at detection year
            h100e_at_detection = h100e_over_time.get(detection_year, 0.0)
        else:
            # Never detected - use final year
            final_year = years[-1]
            h100e_at_detection = h100e_over_time.get(final_year, 0.0)
            detection_year = final_year

        compute_at_detection.append(h100e_at_detection)

        # Calculate operational time
        construction_start = covert_fab.construction_start_year
        construction_duration = covert_fab.construction_duration
        operational_start = construction_start + construction_duration
        operational_time = max(0.0, detection_year - operational_start)
        operational_time_at_detection.append(operational_time)

    return compute_at_detection, operational_time_at_detection


def calculate_ccdf(values):
    """Calculate complementary cumulative distribution function (CCDF).

    Creates one point for every simulation where the fab was built.
    """
    if not values:
        return []

    values_sorted = np.sort(values)
    total_count = len(values_sorted)
    x_values = []
    ccdf_y = []

    for i, x in enumerate(values_sorted):
        # CCDF at x = (number of values > x) / total
        num_greater = total_count - (i + 1)
        ccdf = num_greater / total_count
        x_values.append(float(x))
        ccdf_y.append(float(ccdf))

    return [{"x": x, "y": y} for x, y in zip(x_values, ccdf_y)]


def extract_fab_detection_statistics(simulation_results, years, detection_thresholds, likelihood_ratios):
    """Extract comprehensive fab detection statistics for multiple thresholds.

    Only includes simulations where a covert fab was built.
    """
    compute_ccdfs = {}
    op_time_ccdfs = {}

    for threshold in detection_thresholds:
        compute_at_detection, operational_time_at_detection = extract_fab_compute_at_detection(
            simulation_results, years, threshold
        )
        # Calculate CCDFs - values list is already filtered to only sims with fab
        compute_ccdfs[threshold] = calculate_ccdf(compute_at_detection)
        op_time_ccdfs[threshold] = calculate_ccdf(operational_time_at_detection)

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

        # Get cumulative compute production from the fab over time
        h100e_over_time = covert_fab.get_cumulative_compute_production_over_time()

        # Use simulation years to calculate detection
        us_beliefs = detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"]
        sim_years = sorted(us_beliefs.p_project_exists_update_history.keys()
                          if us_beliefs.p_project_exists_update_history else [])

        detection_year = calculate_fab_detection_year(covert_fab, sim_years, threshold)

        if detection_year is not None:
            # Get compute at detection year
            h100e_at_detection = h100e_over_time.get(detection_year, 0.0)
        else:
            # Never detected - use final year
            final_year = years[-1]
            h100e_at_detection = h100e_over_time.get(final_year, 0.0)
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

    for lr in likelihood_ratios:
        h100_years_at_threshold = []

        for covert_projects, detectors in simulation_results:
            us_beliefs = detectors["us_intelligence"].beliefs_about_projects["prc_covert_project"]
            sim_years = sorted(us_beliefs.p_project_exists_update_history.keys()
                             if us_beliefs.p_project_exists_update_history else [])

            detection_year = calculate_project_detection_year(us_beliefs, sim_years, lr, p_project_exists)
            h100_years = calculate_h100_years_until_detection(covert_projects, sim_years, agreement_year, detection_year)
            h100_years_at_threshold.append(h100_years)

        project_h100_years_ccdfs[lr] = calculate_ccdf(h100_years_at_threshold)

    return project_h100_years_ccdfs


def extract_individual_project_detection_data(simulation_results, agreement_year, threshold, app_params):
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
# UTILITY - TYPE CONVERSION
# =============================================================================

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# =============================================================================
# MAIN EXTRACTION FUNCTION
# =============================================================================

def extract_plot_data(model, app_params):
    """
    Extract all plot data from model simulation results.

    Args:
        model: Model instance with simulation results
        app_params: Application parameters

    Returns:
        Dictionary containing all plot data
    """
    likelihood_ratios = LIKELIHOOD_RATIOS
    p_project_exists = app_params.covert_project_parameters.p_project_exists
    if not model.simulation_results:
        return {"error": "No simulation results"}

    agreement_year = app_params.simulation_settings.start_year
    years = get_years_from_simulation(model.simulation_results)
    years_array = np.array(years)

    # Track which simulations have fabs built
    fab_built_in_sim = [
        cp["prc_covert_project"].covert_fab is not None
        for cp, _ in model.simulation_results
    ]

    # Extract time series data (needed for calculations)
    project_beliefs_by_sim = extract_project_beliefs_over_time(model.simulation_results, years)
    h100e_by_sim = extract_fab_production_over_time(model.simulation_results, years)
    survival_rate_by_sim = extract_survival_rates_over_time(model.simulation_results, years)
    dark_compute_by_sim = extract_dark_compute_over_time(model.simulation_results, years)
    operational_dark_compute_by_sim = extract_operational_dark_compute_over_time(model.simulation_results, years)
    datacenter_capacity_by_sim = extract_datacenter_capacity_over_time(model.simulation_results, years, agreement_year)
    lr_datacenters_by_sim = extract_datacenter_lr_over_time(model.simulation_results, years, agreement_year)
    h100_years_by_sim = extract_h100_years_over_time(model.simulation_results, years, agreement_year)
    cumulative_lr_by_sim = extract_cumulative_lr_over_time(model.simulation_results, years)
    lr_initial_by_sim, lr_sme_by_sim, lr_other_by_sim = extract_project_lr_components_over_time(model.simulation_results, years)

    # Extract fab-specific data
    lr_inventory_by_sim, lr_procurement_by_sim, lr_fab_other_by_sim = extract_fab_lr_components_over_time(model.simulation_results, years)
    lr_fab_combined_by_sim = extract_fab_combined_lr_over_time(model.simulation_results, years)
    is_operational_by_sim = extract_fab_operational_status_over_time(model.simulation_results, years)
    fab_params = extract_fab_production_parameters(model.simulation_results, years, agreement_year)
    watts_per_tpp_by_sim = calculate_watts_per_tpp_by_sim(fab_params['transistor_density'])
    energy_by_source_array, source_labels = extract_energy_breakdown_by_source(model.simulation_results, years)

    # Detection data
    # Use likelihood ratios as detection thresholds
    compute_ccdfs, op_time_ccdfs = extract_fab_detection_statistics(
        model.simulation_results, years, likelihood_ratios, likelihood_ratios
    )
    # Use LR=5 as the dashboard threshold (matches default in frontend)
    dashboard_lr_threshold = 5
    individual_fab_h100e, individual_fab_time, individual_fab_process_nodes, individual_fab_energy = \
        extract_individual_fab_detection_data(model.simulation_results, years, dashboard_lr_threshold, app_params)
    individual_project_h100e, individual_project_energy, individual_project_time, individual_project_h100_years = \
        extract_individual_project_detection_data(model.simulation_results, agreement_year, dashboard_lr_threshold, app_params)

    # Extract initial stock data
    diversion_proportion = app_params.covert_project_strategy.proportion_of_initial_compute_stock_to_divert
    initial_stock_data = extract_initial_stock_data(model.simulation_results, likelihood_ratios, diversion_proportion)

    # Helper to format percentiles for output (p25, median, p75, individual)
    def fmt_pct(data, scale_fn=None):
        """Calculate and format percentiles plus individual simulation data."""
        p = calculate_percentiles(np.array(data))
        if scale_fn:
            return {
                "p25": scale_fn(p[0]).tolist(),
                "median": scale_fn(p[1]).tolist(),
                "p75": scale_fn(p[2]).tolist(),
                "individual": [list(scale_fn(np.array(sim))) for sim in data]
            }
        return {
            "p25": p[0].tolist(),
            "median": p[1].tolist(),
            "p75": p[2].tolist(),
            "individual": [list(sim) for sim in data]
        }

    # Build results dictionary with structure for main sections
    results = {
        # =====================================================================
        # GLOBAL SIMULATION PARAMETERS
        # =====================================================================
        "num_simulations": len(model.simulation_results),
        "prob_fab_built": sum(fab_built_in_sim) / len(fab_built_in_sim) if len(fab_built_in_sim) > 0 else 0.0,
        "p_project_exists": p_project_exists,

        # =====================================================================
        # DARK COMPUTE MODEL (main page overview)
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
            "h100_years_ccdf": extract_project_h100_years_ccdfs(
                model.simulation_results, years, agreement_year, likelihood_ratios, p_project_exists
            ),

            # -------------------------------------------------------------------
            # H100-Years and Evidence Over Time Plot
            # -------------------------------------------------------------------
            "h100_years": fmt_pct(h100_years_by_sim),
            "cumulative_lr": fmt_pct(cumulative_lr_by_sim),

            # -------------------------------------------------------------------
            # Dark Compute Stock Breakdown
            # -------------------------------------------------------------------
            "initial_dark_compute": fmt_pct(dark_compute_by_sim, convert_to_thousands),
            "covert_fab_flow": fmt_pct(filter_to_with_fab(h100e_by_sim, fab_built_in_sim)),  # Keep in raw H100e units to match monthly production
            "survival_rate": fmt_pct(survival_rate_by_sim),
            "total_dark_compute": fmt_pct(dark_compute_by_sim, convert_to_thousands),

            # -------------------------------------------------------------------
            # Data Center Capacity Breakdown
            # -------------------------------------------------------------------
            "datacenter_capacity": fmt_pct(datacenter_capacity_by_sim),
            "dark_compute_energy": energy_by_source_array.tolist(),
            "energy_source_labels": source_labels,
            "operational_dark_compute": fmt_pct(operational_dark_compute_by_sim, convert_to_thousands),

            # -------------------------------------------------------------------
            # Strength of Evidence Breakdown
            # -------------------------------------------------------------------
            "lr_initial_stock": fmt_pct(lr_initial_by_sim),
            "lr_diverted_sme": fmt_pct(lr_sme_by_sim),
            "lr_other_intel": fmt_pct(lr_other_by_sim),
            "posterior_prob_project": fmt_pct(project_beliefs_by_sim),

            # -------------------------------------------------------------------
            # Covert Project Dashboard - Individual simulation data
            # -------------------------------------------------------------------
            "individual_project_h100e_before_detection": individual_project_h100e,
            "individual_project_energy_before_detection": individual_project_energy,
            "individual_project_time_before_detection": individual_project_time,
            "individual_project_h100_years_before_detection": individual_project_h100_years,
        },

        # =====================================================================
        # INITIAL DARK COMPUTE (diverted chips section)
        # =====================================================================
        "initial_dark_compute": {
            "years": years_array.tolist(),
            # Covert fab production flow (cumulative over time)
            "h100e": fmt_pct(filter_to_with_fab(h100e_by_sim, fab_built_in_sim), convert_to_thousands),
            # Chip survival rate over time
            "survival_rate": fmt_pct(survival_rate_by_sim),
            # Total dark compute (surviving, not capacity-limited)
            "dark_compute": fmt_pct(dark_compute_by_sim, convert_to_thousands),
        },

        # =====================================================================
        # COVERT DATA CENTERS
        # =====================================================================
        "covert_datacenters": {
            "years": years_array.tolist(),
            # Datacenter capacity (GW)
            "datacenter_capacity": fmt_pct(datacenter_capacity_by_sim),
            # Energy consumption by source (stacked area plot)
            "energy_by_source": energy_by_source_array.tolist(),
            "source_labels": source_labels,
            # Operational dark compute (capacity-limited)
            "operational_dark_compute": fmt_pct(operational_dark_compute_by_sim, convert_to_thousands),
            # Datacenter detection
            "lr_datacenters": fmt_pct(lr_datacenters_by_sim),
            "datacenter_detection_prob": (np.mean(np.array(lr_datacenters_by_sim) >= 5.0, axis=0)).tolist(),
        },

        # =====================================================================
        # COVERT FAB
        # =====================================================================
        "covert_fab": {
            "years": years_array.tolist(),
            # Detection - CCDF plots and thresholds
            "compute_ccdf": compute_ccdfs.get(dashboard_lr_threshold, []),
            "compute_ccdfs": compute_ccdfs,
            "op_time_ccdf": op_time_ccdfs.get(dashboard_lr_threshold, []),
            "op_time_ccdfs": op_time_ccdfs,
            "likelihood_ratios": likelihood_ratios,
            # Detection - LR component breakdown (inventory, procurement, other)
            "lr_inventory": fmt_pct(filter_to_with_fab(lr_inventory_by_sim, fab_built_in_sim)),
            "lr_procurement": fmt_pct(filter_to_with_fab(lr_procurement_by_sim, fab_built_in_sim)),
            "lr_other": fmt_pct(filter_to_with_fab(lr_fab_other_by_sim, fab_built_in_sim)),
            # Combined LR (product of inventory, procurement, and other) - from fab's cumulative_detection_likelihood_ratio method
            "lr_combined": fmt_pct(filter_to_with_fab(lr_fab_combined_by_sim, fab_built_in_sim)),
            # Production - Compute factors breakdown
            # Calculate proportion operational (not percentiles)
            "is_operational": (lambda: (
                filtered := filter_to_with_fab(is_operational_by_sim, fab_built_in_sim),
                proportion := np.mean(np.array(filtered), axis=0).tolist(),
                {
                    "proportion": proportion,
                    "individual": [list(sim) for sim in filtered]
                }
            )[-1])(),
            "wafer_starts": fmt_pct(filter_to_with_fab(fab_params['wafer_starts'], fab_built_in_sim)),
            "chips_per_wafer": fmt_pct(filter_to_with_fab(fab_params['chips_per_wafer'], fab_built_in_sim)),
            "architecture_efficiency": fmt_pct(filter_to_with_fab(fab_params['architecture_efficiency'], fab_built_in_sim)),
            "compute_per_wafer_2022_arch": fmt_pct(filter_to_with_fab(fab_params['compute_per_wafer_2022_arch'], fab_built_in_sim)),
            "transistor_density": fmt_pct(filter_to_with_fab(fab_params['transistor_density'], fab_built_in_sim)),
            "watts_per_tpp": fmt_pct(filter_to_with_fab(watts_per_tpp_by_sim, fab_built_in_sim)),
            "process_node_by_sim": filter_to_with_fab(fab_params['process_node'], fab_built_in_sim),
            # Architecture efficiency and watts per TPP
            "architecture_efficiency_at_agreement": estimate_architecture_efficiency_relative_to_h100(agreement_year, agreement_year),
            "watts_per_tpp_curve": PRCCovertFab.watts_per_tpp_relative_to_H100(),
            # Dashboard - Individual simulation data
            "individual_h100e_before_detection": individual_fab_h100e,
            "individual_time_before_detection": individual_fab_time,
            "individual_process_node": individual_fab_process_nodes,
            "individual_energy_before_detection": individual_fab_energy,
            "fab_built": fab_built_in_sim,
        },

        # =====================================================================
        # INITIAL DARK COMPUTE STOCK (diverted chips)
        # =====================================================================
        "initial_stock": initial_stock_data,
    }

    # Convert numpy types to native Python types for JSON serialization
    converted_results = convert_numpy_types(results)

    # Debug: Print covert fab CCDF data
    print("\n" + "="*80)
    print("COVERT FAB COMPUTE CCDFs:")
    print(f"Likelihood ratios: {converted_results['covert_fab']['likelihood_ratios']}")
    print(f"Compute CCDFs keys: {list(converted_results['covert_fab']['compute_ccdfs'].keys())}")
    for lr, ccdf_data in converted_results['covert_fab']['compute_ccdfs'].items():
        print(f"\nLR={lr}: {len(ccdf_data)} data points")
        if ccdf_data:
            print(f"  First 3 points: {ccdf_data[:3]}")
            print(f"  Last 3 points: {ccdf_data[-3:]}")
    print("="*80 + "\n")

    return converted_results
