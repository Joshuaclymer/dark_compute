"""
Clean, structured plot data extraction for the covert compute production model.

This module provides a well-organized alternative to the extract_plot_data function
in app.py, with clear separation of concerns and minimal redundancy.
"""

import numpy as np
import csv
import os
from typing import List, Tuple, Dict, Any
from black_project_backend.classes.black_project_stock import H100_TPP_PER_CHIP, H100_WATTS_PER_TPP
from black_project_backend.classes.largest_ai_project import LargestAIProject


# =============================================================================
# GLOBAL COMPUTE PRODUCTION (loaded from global_compute_production.csv)
# =============================================================================

# Cache for global compute production data
_cached_global_compute_production = None

def _parse_compute_value(s):
    """Parse a compute value string with optional K/M/B/T suffix."""
    if not s:
        return None
    s = s.strip().replace(',', '')
    multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            try:
                return float(s[:-1]) * mult
            except ValueError:
                return None
    try:
        return float(s)
    except ValueError:
        return None


def _load_global_compute_production():
    """Load the global compute production data from global_compute_production.csv.

    Returns a dict with:
        - years: list of years
        - cagr_stock: list of CAGR values for overall compute stock (H100e)
        - compute_added: list of compute added values (H100e) per year
        - total_stock: list of total H100e in world (no decay)
    """
    global _cached_global_compute_production
    if _cached_global_compute_production is not None:
        return _cached_global_compute_production

    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'global_compute_production.csv'
    )

    years = []
    cagr_stock = []
    compute_added = []
    total_stock = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        # Skip header rows (first 3 rows: header + 2 metadata rows)
        for _ in range(3):
            next(reader)

        for row in reader:
            if len(row) > 0 and row[0]:
                try:
                    year = int(row[0])
                    # Column F (index 5) is "CAGR overall compute stock (H100e)"
                    cagr = float(row[5]) if row[5] else None
                    # Column T (index 19) is "Compute added (H100e)"
                    added = _parse_compute_value(row[19]) if len(row) > 19 else None
                    # Column U (index 20) is "H100e in world, no decay"
                    stock = _parse_compute_value(row[20]) if len(row) > 20 else None

                    years.append(year)
                    cagr_stock.append(cagr)
                    compute_added.append(added)
                    total_stock.append(stock)
                except (ValueError, IndexError):
                    continue

    _cached_global_compute_production = {
        'years': years,
        'cagr_stock': cagr_stock,
        'compute_added': compute_added,
        'total_stock': total_stock
    }
    return _cached_global_compute_production


def get_global_compute_stock(year: float) -> float:
    """Get the global compute stock (H100e) for a given year.

    Uses the "H100e in world, no decay" column from global_compute_production.csv.
    Linearly interpolates between years.
    """
    data = _load_global_compute_production()

    # Filter to valid entries
    valid_years = []
    valid_stocks = []
    for y, s in zip(data['years'], data['total_stock']):
        if s is not None:
            valid_years.append(y)
            valid_stocks.append(s)

    if not valid_years:
        return 0.0

    return float(np.interp(year, valid_years, valid_stocks))


def get_global_compute_production_between_years(start_year: float, end_year: float) -> float:
    """Calculate total global compute production between two years.

    Uses the change in global compute stock (H100e in world, no decay) between years.
    Production = Stock(end_year) - Stock(start_year)
    """
    start_stock = get_global_compute_stock(start_year)
    end_stock = get_global_compute_stock(end_year)
    return max(0.0, end_stock - start_stock)


from black_project_backend.classes.black_fab import (
    estimate_architecture_efficiency_relative_to_h100,
    predict_watts_per_tpp_from_transistor_density,
    H100_TRANSISTOR_DENSITY_M_PER_MM2,
    PRCBlackFab
)

# Configure likelihood ratio thresholds for detection plots
# Update this list to change all plots automatically
LIKELIHOOD_RATIOS = [1, 2, 4]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def filter_simulations_with_fab(simulation_results):
    """Filter simulations to only include those where a covert fab was built."""
    return [
        black_projects
        for black_projects in simulation_results
        if black_projects["prc_black_project"].black_fab is not None
    ]


def filter_to_with_fab(data_by_sim, fab_built_mask):
    """Filter simulation data to only include simulations where a fab was built."""
    return [sim for i, sim in enumerate(data_by_sim) if fab_built_mask[i]]


def get_years_from_simulation(simulation_results):
    """Extract the common years array from simulation results."""
    if not simulation_results:
        return []

    black_projects = simulation_results[0]
    # Get years from the black project's years list
    return black_projects["prc_black_project"].years


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

def extract_fab_production_over_time(simulation_results, years):
    """Extract H100e production from covert fab over time for each simulation."""
    h100e_by_sim = []

    for black_projects in simulation_results:
        black_fab = black_projects["prc_black_project"].black_fab

        if black_fab is not None:
            # Get cumulative production for all recorded years
            cumulative_by_year = black_fab.get_cumulative_compute_production_over_time()
            # Map to the requested years array
            h100e_counts = [cumulative_by_year.get(year, 0.0) for year in years]
        else:
            h100e_counts = [0.0 for _ in years]

        h100e_by_sim.append(h100e_counts)

    return h100e_by_sim


def extract_survival_rates_over_time(simulation_results, years):
    """Extract chip survival rates over time for each simulation."""
    survival_rate_by_sim = []

    for black_projects in simulation_results:
        black_project_stock = black_projects["prc_black_project"].black_project_stock
        survival_rates = []

        for year in years:
            # Use black_project_energy_by_source for consistency
            initial_energy, fab_energy, initial_h100e, fab_h100e = black_project_stock.black_project_energy_by_source(year)
            surviving = initial_h100e + fab_h100e

            total_compute = black_project_stock.black_project_dead_and_alive(year)
            total = total_compute.total_h100e_tpp()

            survival_rates.append(surviving / total if total > 0 else 0.0)

        survival_rate_by_sim.append(survival_rates)

    return survival_rate_by_sim


def extract_black_project_over_time(simulation_results, years):
    """Extract dark compute (surviving, not limited by capacity) over time."""
    black_project_by_sim = []

    for black_projects in simulation_results:
        black_project_stock = black_projects["prc_black_project"].black_project_stock
        black_project = []

        for year in years:
            # Use black_project_energy_by_source for consistency
            initial_energy, fab_energy, initial_h100e, fab_h100e = black_project_stock.black_project_energy_by_source(year)
            black_project.append(initial_h100e + fab_h100e)

        black_project_by_sim.append(black_project)

    return black_project_by_sim


def extract_operational_black_project_over_time(simulation_results, years):
    """Extract operational dark compute (limited by datacenter capacity) over time."""
    operational_black_project_by_sim = []

    for black_projects in simulation_results:
        operational_black_project = []

        for year in years:
            operational_compute = black_projects["prc_black_project"].operational_black_project(year)
            operational_black_project.append(operational_compute.total_h100e_tpp())

        operational_black_project_by_sim.append(operational_black_project)

    return operational_black_project_by_sim


def extract_datacenter_capacity_over_time(simulation_results, years, agreement_year):
    """Extract total datacenter capacity in GW over time (concealed + unconcealed)."""
    datacenter_capacity_by_sim = []

    for black_projects in simulation_results:
        black_datacenters = black_projects["prc_black_project"].black_datacenters
        datacenter_capacity_gw = []

        for year in years:
            relative_year = year - agreement_year
            capacity_gw = black_datacenters.get_covert_GW_capacity_total(relative_year)
            datacenter_capacity_gw.append(capacity_gw)

        datacenter_capacity_by_sim.append(datacenter_capacity_gw)

    return datacenter_capacity_by_sim


def calculate_prc_datacenter_capacity_trajectory(app_params):
    """Calculate PRC datacenter capacity trajectory from 2025 to agreement year.

    This calculates the total energy consumption of PRC compute stock over time,
    using median parameter values (not sampled).

    Returns:
        dict with:
            - years: list of years from 2025 to agreement_year
            - capacity_gw: list of capacity values in GW for each year
            - capacity_at_agreement_year_gw: capacity at agreement year in GW
    """
    agreement_year = app_params.simulation_settings.agreement_start_year
    exogenous = app_params.black_project_parameters.exogenous_trends

    # Use median growth rate
    growth_rate = exogenous.annual_growth_rate_of_prc_compute_stock_p50
    total_prc_compute_stock_2025 = exogenous.total_prc_compute_stock_in_2025
    energy_efficiency = exogenous.energy_efficiency_of_prc_stock_relative_to_state_of_the_art
    energy_efficiency_improvement = exogenous.improvement_in_energy_efficiency_per_year

    # Create LargestAIProject to get state-of-art efficiency
    largest_ai_project = LargestAIProject(exogenous)

    years = list(range(2025, int(agreement_year) + 1))
    capacity_gw = []

    for year in years:
        years_since_2025 = year - 2025
        # Project compute stock forward
        compute_stock = total_prc_compute_stock_2025 * (growth_rate ** years_since_2025)

        # Get state-of-art efficiency at this year
        state_of_art_efficiency = largest_ai_project.get_energy_efficiency_relative_to_h100(year)
        effective_efficiency = energy_efficiency * state_of_art_efficiency

        # Calculate energy consumption in GW
        # Using the same formula as get_energy_consumption_of_prc_stock_gw
        energy_watts = compute_stock * H100_TPP_PER_CHIP * H100_WATTS_PER_TPP / effective_efficiency
        energy_gw = energy_watts / 1e9
        capacity_gw.append(energy_gw)

    return {
        "prc_capacity_years": years,
        "prc_capacity_gw": capacity_gw,
        "prc_capacity_at_agreement_year_gw": capacity_gw[-1] if capacity_gw else 0.0
    }


def extract_datacenter_lr_over_time(simulation_results, years, agreement_year):
    """Extract likelihood ratio from datacenters over time."""
    lr_datacenters_by_sim = []

    for black_projects in simulation_results:
        black_datacenters = black_projects["prc_black_project"].black_datacenters
        lr_datacenters_over_time = []

        for year in years:
            relative_year = year - agreement_year
            lr_datacenters = black_datacenters.cumulative_lr_from_concealed_datacenters(relative_year)
            lr_datacenters_over_time.append(lr_datacenters)

        lr_datacenters_by_sim.append(lr_datacenters_over_time)

    return lr_datacenters_by_sim


def extract_h100_years_over_time(simulation_results, years, agreement_year):
    """Extract cumulative H100-years over time for each simulation."""
    h100_years_by_sim = []

    for black_projects in simulation_results:
        h100_years_over_time = []
        cumulative_h100_years = 0.0

        for year_idx, year in enumerate(years):
            if year_idx > 0:
                prev_year = years[year_idx - 1]
                time_increment = year - prev_year

                operational_compute = black_projects["prc_black_project"].operational_black_project(prev_year)
                h100e_at_prev_year = operational_compute.total_h100e_tpp()

                cumulative_h100_years += h100e_at_prev_year * time_increment

            h100_years_over_time.append(cumulative_h100_years)

        h100_years_by_sim.append(h100_years_over_time)

    return h100_years_by_sim


def extract_cumulative_lr_over_time(simulation_results, years):
    """Extract cumulative likelihood ratio over time for each simulation."""
    cumulative_lr_by_sim = []

    for black_projects in simulation_results:
        cumulative_lr_over_time = []

        for year in years:
            project_lr = black_projects["prc_black_project"].get_cumulative_evidence_of_black_project(year)
            cumulative_lr_over_time.append(project_lr)

        cumulative_lr_by_sim.append(cumulative_lr_over_time)

    return cumulative_lr_by_sim


def extract_posterior_prob_over_time(cumulative_lr_by_sim, p_project_exists):
    """Compute posterior probability of project existence from cumulative LR and prior.

    Uses Bayes' theorem:
    posterior_odds = prior_odds * likelihood_ratio
    posterior_prob = posterior_odds / (1 + posterior_odds)

    Args:
        cumulative_lr_by_sim: List of lists, cumulative LR over time for each simulation
        p_project_exists: Prior probability that the project exists

    Returns:
        List of lists, posterior probability over time for each simulation
    """
    prior_odds = p_project_exists / (1 - p_project_exists) if p_project_exists < 1 else float('inf')

    posterior_prob_by_sim = []
    for lr_over_time in cumulative_lr_by_sim:
        posterior_over_time = []
        for lr in lr_over_time:
            posterior_odds = prior_odds * lr
            posterior_prob = posterior_odds / (1 + posterior_odds) if posterior_odds < float('inf') else 1.0
            posterior_over_time.append(posterior_prob)
        posterior_prob_by_sim.append(posterior_over_time)

    return posterior_prob_by_sim


def extract_project_lr_components_over_time(simulation_results, years):
    """Extract likelihood ratio components for covert project detection (initial, SME, other intel)."""
    lr_initial_by_sim = []
    lr_sme_by_sim = []
    lr_other_by_sim = []

    for black_projects in simulation_results:
        # Initial stock LR is constant over time
        lr_initial = black_projects["prc_black_project"].get_lr_initial()
        lr_initial_over_time = [lr_initial for _ in years]

        # SME LR (from fab procurement/inventory) is constant over time
        lr_sme = black_projects["prc_black_project"].get_lr_sme()
        lr_sme_over_time = [lr_sme for _ in years]

        # Other intelligence (workers, etc.) varies over time
        lr_other_over_time = []
        for year in years:
            lr_other_over_time.append(
                black_projects["prc_black_project"].get_lr_other(year)
            )

        lr_initial_by_sim.append(lr_initial_over_time)
        lr_sme_by_sim.append(lr_sme_over_time)
        lr_other_by_sim.append(lr_other_over_time)

    return lr_initial_by_sim, lr_sme_by_sim, lr_other_by_sim


def extract_detailed_lr_components_over_time(simulation_results, years):
    """Extract detailed breakdown of LR components for display."""
    lr_prc_accounting_by_sim = []
    lr_sme_inventory_by_sim = []

    for black_projects in simulation_results:
        project = black_projects["prc_black_project"]

        # Get initial stock LR components (constant over time)
        lr_prc = project.black_project_stock.lr_from_prc_compute_accounting
        lr_prc_accounting_by_sim.append([lr_prc for _ in years])

        # Get SME LR components (constant over time)
        if project.black_fab is not None:
            lr_inventory = project.black_fab.lr_inventory
        else:
            lr_inventory = 1.0
        lr_sme_inventory_by_sim.append([lr_inventory for _ in years])

    return lr_prc_accounting_by_sim, lr_sme_inventory_by_sim


def extract_fab_combined_lr_over_time(simulation_results, years):
    """Extract cumulative combined likelihood ratio from fab's cumulative_detection_likelihood_ratio method."""
    combined_lr_by_sim = []

    for black_projects in simulation_results:
        black_fab = black_projects["prc_black_project"].black_fab

        if black_fab is not None:
            # Call the fab's cumulative_detection_likelihood_ratio method for each year
            combined_lr_over_time = [
                black_fab.cumulative_detection_likelihood_ratio(year)
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

    for black_projects in simulation_results:
        lr_inventory_over_time = []
        lr_procurement_over_time = []
        lr_other_over_time = []

        for year in years:
            lr_inventory_over_time.append(
                black_projects["prc_black_project"].get_fab_lr_inventory(year)
            )
            lr_procurement_over_time.append(
                black_projects["prc_black_project"].get_fab_lr_procurement(year)
            )
            lr_other_over_time.append(
                black_projects["prc_black_project"].get_fab_lr_other(year)
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

    for black_projects in simulation_results:
        is_operational = []

        for year in years:
            is_operational.append(
                black_projects["prc_black_project"].get_fab_is_operational(year)
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

    for black_projects in simulation_results:
        wafer_starts = []
        architecture_efficiency = []
        compute_per_wafer_2022_arch = []

        for year in years:
            wafer_starts.append(
                black_projects["prc_black_project"].get_fab_wafer_starts_per_month()
            )

            arch_efficiency = estimate_architecture_efficiency_relative_to_h100(year, agreement_year)
            architecture_efficiency.append(arch_efficiency)

            chips_per_wafer = black_projects["prc_black_project"].get_fab_h100_sized_chips_per_wafer()
            transistor_density = black_projects["prc_black_project"].get_fab_transistor_density_relative_to_h100()
            compute_per_wafer_2022_arch.append(chips_per_wafer * transistor_density)

        wafer_starts_by_sim.append(wafer_starts)
        chips_per_wafer_by_sim.append(
            [black_projects["prc_black_project"].get_fab_h100_sized_chips_per_wafer()] * len(years)
        )
        architecture_efficiency_by_sim.append(architecture_efficiency)
        compute_per_wafer_2022_arch_by_sim.append(compute_per_wafer_2022_arch)
        transistor_density_by_sim.append(
            [black_projects["prc_black_project"].get_fab_transistor_density_relative_to_h100()] * len(years)
        )
        process_node_by_sim.append(
            black_projects["prc_black_project"].get_fab_process_node()
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

    # Collect fab energy efficiency from each simulation that has a covert fab
    fab_efficiency_list = []
    for sim_idx, black_projects in enumerate(simulation_results):
        black_fab = black_projects["prc_black_project"].black_fab
        if black_fab and black_fab.black_project_monthly_production_rate_history:
            # Get any year from the production history (fab produces one type per simulation)
            sample_year = list(black_fab.black_project_monthly_production_rate_history.keys())[0]
            compute_obj = black_fab.black_project_monthly_production_rate_history[sample_year]

            # Get the single chip from the Compute object
            if compute_obj.chip_counts:
                chip = list(compute_obj.chip_counts.keys())[0]
                # Calculate energy_efficiency_relative_to_h100 from chip properties
                # energy_efficiency = (h100e_tpp_per_chip * H100_TPP_PER_CHIP * H100_WATTS_PER_TPP) / W_of_energy_consumed
                fab_efficiency = (chip.h100e_tpp_per_chip * H100_TPP_PER_CHIP * H100_WATTS_PER_TPP) / chip.W_of_energy_consumed
                fab_efficiency_list.append(fab_efficiency)

    for sim_idx, black_projects in enumerate(simulation_results):
        black_project_stock = black_projects["prc_black_project"].black_project_stock

        for year_idx, year in enumerate(years):
            initial_energy, fab_energy, initial_h100e, fab_h100e = \
                black_project_stock.black_project_energy_by_source(year)
            initial_energy_all_sims[sim_idx, year_idx] = initial_energy
            fab_energy_all_sims[sim_idx, year_idx] = fab_energy
            initial_h100e_all_sims[sim_idx, year_idx] = initial_h100e
            fab_h100e_all_sims[sim_idx, year_idx] = fab_h100e

    # Compute median across simulations
    initial_energy_median = np.median(initial_energy_all_sims, axis=0)
    fab_energy_median = np.median(fab_energy_all_sims, axis=0)
    initial_h100e_median = np.median(initial_h100e_all_sims, axis=0)
    fab_h100e_median = np.median(fab_h100e_all_sims, axis=0)

    # Calculate average efficiency for initial stock over all years
    initial_energy_total = np.sum(initial_energy_median)
    initial_h100e_total = np.sum(initial_h100e_median)

    initial_efficiency = (initial_h100e_total / initial_energy_total) if initial_energy_total > 0 else 0

    # Calculate baseline H100 efficiency
    h100_baseline_efficiency = 1e9 / (H100_TPP_PER_CHIP * H100_WATTS_PER_TPP)

    initial_efficiency_relative = initial_efficiency / h100_baseline_efficiency if h100_baseline_efficiency > 0 else 0

    # Take median of fab efficiency across simulations with fabs
    if fab_efficiency_list:
        fab_efficiency_relative = np.median(fab_efficiency_list)
    else:
        fab_efficiency_relative = 1.0  # Fallback if no fab was built

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

def extract_prc_compute_over_time(years, initial_compute_parameters):
    """Extract PRC compute stock over time based on growth parameters.

    This calculates the expected PRC compute stock for each year based on
    the 2025 baseline and annual growth rate. Each sample uses a single
    sampled growth rate applied consistently across all years.
    """
    from black_project_backend.classes.black_project_stock import sample_prc_growth_rate, compute_prc_compute_stock

    # We'll generate multiple samples to get percentiles
    num_samples = 100
    prc_compute_by_year_by_sample = []

    for _ in range(num_samples):
        # Sample growth rate once per simulation using metalog distribution
        growth_rate = sample_prc_growth_rate(initial_compute_parameters)

        # Apply this growth rate consistently across all years
        prc_compute_over_time = []
        for year in years:
            prc_stock = compute_prc_compute_stock(year, growth_rate, initial_compute_parameters)
            prc_compute_over_time.append(prc_stock)
        prc_compute_by_year_by_sample.append(prc_compute_over_time)

    return prc_compute_by_year_by_sample


def extract_largest_company_compute_over_time(years, largest_ai_project: LargestAIProject):
    """Extract largest AI project compute over time.

    Returns compute stock values in H100-equivalent units, calculated using
    exponential growth from the 2025 baseline.
    """
    return largest_ai_project.get_compute_trajectory(years)


def interpolate_domestic_proportion(year, initial_compute_parameters):
    """Linearly interpolate the proportion of PRC chips produced domestically for a given year.

    Interpolates between 2026 and 2030 values. Returns:
    - proportion_2026 for years <= 2026
    - proportion_2030 for years >= 2030
    - Linear interpolation for years between 2026 and 2030
    """
    prop_2026 = initial_compute_parameters.proportion_of_prc_chip_stock_produced_domestically_2026
    prop_2030 = initial_compute_parameters.proportion_of_prc_chip_stock_produced_domestically_2030

    if year <= 2026:
        return prop_2026
    elif year >= 2030:
        return prop_2030
    else:
        # Linear interpolation between 2026 and 2030
        t = (year - 2026) / (2030 - 2026)
        return prop_2026 + t * (prop_2030 - prop_2026)


def extract_initial_stock_data(simulation_results, likelihood_ratios, diversion_proportion):
    """Extract initial dark compute stock data from simulation results."""
    initial_prc_stock_samples = []
    initial_compute_samples = []
    lr_prc_accounting_samples = []

    for black_projects in simulation_results:
        black_project_stock = black_projects["prc_black_project"].black_project_stock

        # Extract initial stock values
        initial_prc_stock_samples.append(black_project_stock.initial_prc_stock)
        initial_compute_samples.append(black_project_stock.initial_prc_black_project)

        # Extract likelihood ratios
        lr_prc = black_project_stock.lr_from_prc_compute_accounting
        lr_prc_accounting_samples.append(lr_prc)

    # Calculate detection probabilities for each threshold
    detection_probs = {}
    for threshold in likelihood_ratios:
        num_detected = sum(1 for lr in lr_prc_accounting_samples if lr >= threshold)
        detection_probs[f"{threshold}x"] = num_detected / len(lr_prc_accounting_samples)

    return {
        'initial_prc_stock_samples': initial_prc_stock_samples,
        'initial_compute_stock_samples': initial_compute_samples,
        'diversion_proportion': diversion_proportion,
        'lr_prc_accounting_samples': lr_prc_accounting_samples,
        'initial_black_project_detection_probs': detection_probs,
    }


# =============================================================================
# DETECTION ANALYSIS
# =============================================================================

def calculate_fab_detection_year(black_fab, years, threshold):
    """Calculate the year when fab is detected based on likelihood ratio threshold.

    Uses the fab's cumulative_detection_likelihood_ratio method which combines:
    - inventory accounting (constant)
    - procurement accounting (constant)
    - other intelligence (grows over time with workers)

    Detection occurs the first year the cumulative LR exceeds the threshold.
    """
    if black_fab is None:
        return None

    for year in years:
        # Get the fab's cumulative LR (inventory × procurement × other) for this year
        year_lr = black_fab.cumulative_detection_likelihood_ratio(year)

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

    for black_projects in simulation_results:
        black_fab = black_projects["prc_black_project"].black_fab

        if black_fab is None:
            continue

        # Get cumulative compute production from the fab over time
        h100e_over_time = black_fab.get_cumulative_compute_production_over_time()

        # Use simulation years to calculate detection
        sim_years = black_projects["prc_black_project"].years

        detection_year = calculate_fab_detection_year(black_fab, sim_years, threshold)

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
        construction_start = black_fab.construction_start_year
        construction_duration = black_fab.construction_duration
        operational_start = construction_start + construction_duration
        operational_time = max(0.0, detection_year - operational_start)
        operational_time_at_detection.append(operational_time)

    return compute_at_detection, operational_time_at_detection


def calculate_ccdf(values):
    """Calculate complementary cumulative distribution function (CCDF).

    Returns points for unique x values only, for smooth linear interpolation.
    For each unique x, the CCDF value is P(X > x) = (count of values > x) / total.
    """
    if not values:
        return []

    values_sorted = np.sort(values)
    total_count = len(values_sorted)

    # Get unique x values and their CCDFs
    # For CCDF at x, we want P(X > x), which is the proportion of values strictly greater than x
    unique_x = []
    ccdf_y = []

    prev_x = None
    for i, x in enumerate(values_sorted):
        if x != prev_x:
            # New unique x value - CCDF is proportion of values > x
            num_greater = total_count - (i + 1)
            ccdf = num_greater / total_count
            unique_x.append(float(x))
            ccdf_y.append(float(ccdf))
            prev_x = x
        # For duplicate x values, just update the last y (will be lower)
        else:
            num_greater = total_count - (i + 1)
            ccdf_y[-1] = num_greater / total_count

    return [{"x": x, "y": y} for x, y in zip(unique_x, ccdf_y)]


def calculate_smoothed_ccdf(values, num_points=100, floor_value=0.0):
    """Calculate a smoothed CCDF using linear interpolation.

    This produces smooth curves for data with discrete/clustered values
    like time measurements that fall on year boundaries.

    Args:
        values: List of observed values (only events that occurred)
        num_points: Number of points to evaluate CCDF at
        floor_value: Minimum CCDF value (proportion of events that never occur).
                     The CCDF will plateau at this value instead of going to 0.
    """
    from scipy import interpolate

    if not values:
        # If no values, return a flat line at floor_value
        return [{"x": 0.0, "y": float(floor_value)}]

    values_arr = np.array(values)

    # Handle case where all values are the same
    if np.all(values_arr == values_arr[0]):
        return [{"x": float(values_arr[0]), "y": float(floor_value)}]

    # Sort values and compute empirical CCDF
    values_sorted = np.sort(values_arr)
    n = len(values_sorted)

    # The CCDF should range from 1.0 at the start to floor_value at the end
    detected_proportion = 1.0 - floor_value

    # Build empirical CCDF points: at each sorted value, CCDF = P(X > x)
    # For the i-th sorted value, there are (n - i - 1) values greater than it
    empirical_x = []
    empirical_y = []

    for i, x in enumerate(values_sorted):
        # CCDF just before this point (number >= x / n)
        ccdf_before = (n - i) / n
        # CCDF just after this point (number > x / n)
        ccdf_after = (n - i - 1) / n

        # Scale to account for floor_value
        y_before = floor_value + detected_proportion * ccdf_before
        y_after = floor_value + detected_proportion * ccdf_after

        # Add point just before the step (if not first point)
        if i == 0 or x != values_sorted[i - 1]:
            empirical_x.append(float(x))
            empirical_y.append(float(y_before))

        # Add point just after the step
        empirical_x.append(float(x))
        empirical_y.append(float(y_after))

    # Remove duplicate x values by averaging y values
    unique_x = []
    unique_y = []
    i = 0
    while i < len(empirical_x):
        x = empirical_x[i]
        # Collect all y values for this x
        y_vals = [empirical_y[i]]
        j = i + 1
        while j < len(empirical_x) and empirical_x[j] == x:
            y_vals.append(empirical_y[j])
            j += 1
        unique_x.append(x)
        unique_y.append(np.mean(y_vals))
        i = j

    # Create linear interpolation function
    if len(unique_x) < 2:
        return [{"x": unique_x[0], "y": unique_y[0]}]

    interp_func = interpolate.interp1d(
        unique_x, unique_y,
        kind='linear',
        bounds_error=False,
        fill_value=(unique_y[0], unique_y[-1])
    )

    # Evaluate at evenly spaced points
    x_min = float(np.min(values_arr))
    x_max = float(np.max(values_arr))
    x_points = np.linspace(x_min, x_max, num_points)
    y_points = interp_func(x_points)

    return [{"x": float(x), "y": float(y)} for x, y in zip(x_points, y_points)]


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

    for black_projects in simulation_results:
        black_fab = black_projects["prc_black_project"].black_fab

        if black_fab is None:
            continue

        # Get cumulative compute production from the fab over time
        h100e_over_time = black_fab.get_cumulative_compute_production_over_time()

        # Use simulation years to calculate detection
        sim_years = black_projects["prc_black_project"].years

        detection_year = calculate_fab_detection_year(black_fab, sim_years, threshold)

        if detection_year is not None:
            # Get compute at detection year
            h100e_at_detection = h100e_over_time.get(detection_year, 0.0)
        else:
            # Never detected - use final year
            final_year = years[-1]
            h100e_at_detection = h100e_over_time.get(final_year, 0.0)
            detection_year = final_year

        construction_start = black_fab.construction_start_year
        construction_duration = black_fab.construction_duration
        operational_start = construction_start + construction_duration
        operational_time = max(0.0, detection_year - operational_start)

        individual_h100e.append(h100e_at_detection)
        individual_time.append(operational_time)
        individual_process_nodes.append(black_fab.process_node.value)

        energy_gw = (h100e_at_detection * H100_TPP_PER_CHIP * H100_WATTS_PER_TPP /
                    app_params.black_project_parameters.exogenous_trends.energy_efficiency_of_prc_stock_relative_to_state_of_the_art / 1e9)
        individual_energy.append(energy_gw)

    return individual_h100e, individual_time, individual_process_nodes, individual_energy


def calculate_project_detection_year(black_project, years, likelihood_ratio):
    """Calculate detection year for covert project based on likelihood ratio threshold.

    Uses get_cumulative_evidence_of_black_project method from PRCBlackProject class.
    """
    if not years:
        return None

    for year in years:
        cumulative_lr = black_project.get_cumulative_evidence_of_black_project(year)

        if cumulative_lr >= likelihood_ratio:
            return year

    return None


def calculate_h100_years_until_detection(black_project, years, detection_year):
    """Calculate cumulative H100-years until detection (or end of simulation).

    Uses h100_years_to_date method from PRCBlackProject class.
    """
    end_year = detection_year if detection_year is not None else max(years)

    # Use the PRCBlackProject's h100_years_to_date method
    h100_years = black_project.h100_years_to_date(end_year, years)

    return h100_years


def extract_project_h100_years_ccdfs(simulation_results, years, agreement_year, likelihood_ratios, p_project_exists):
    """Calculate CCDFs for H100-years at different detection thresholds for the covert project."""
    project_h100_years_ccdfs = {}

    for lr in likelihood_ratios:
        h100_years_at_threshold = []

        for black_projects in simulation_results:
            black_project = black_projects["prc_black_project"]

            # Use the covert project's years list
            sim_years = black_project.years

            detection_year = calculate_project_detection_year(black_project, sim_years, lr)
            h100_years = calculate_h100_years_until_detection(black_project, sim_years, detection_year)
            h100_years_at_threshold.append(h100_years)

        project_h100_years_ccdfs[lr] = calculate_ccdf(h100_years_at_threshold)

    return project_h100_years_ccdfs


def extract_average_covert_compute_ccdfs(simulation_results, years, agreement_year, likelihood_ratios):
    """Calculate CCDFs for average covert compute operated from agreement to detection.

    For each simulation and detection threshold:
    1. Calculate detection year based on likelihood ratio
    2. Compute average operational dark compute (Operating AI chips) from agreement year to detection
    3. Return CCDF of these averages

    Average is calculated as: total H100-years / time duration = time-weighted average operational compute
    """
    average_compute_ccdfs = {}

    for lr in likelihood_ratios:
        average_compute_at_threshold = []

        for black_projects in simulation_results:
            black_project = black_projects["prc_black_project"]
            sim_years = black_project.years

            # Calculate detection year for this simulation
            detection_year = calculate_project_detection_year(black_project, sim_years, lr)
            if detection_year is None:
                detection_year = max(sim_years)

            # Calculate time duration from agreement to detection
            time_duration = detection_year - agreement_year
            if time_duration <= 0:
                # If detection at or before agreement, average is 0
                average_compute_at_threshold.append(0.0)
                continue

            # Calculate total H100-years from agreement to detection
            h100_years = calculate_h100_years_until_detection(black_project, sim_years, detection_year)

            # Average operational compute = total H100-years / time duration
            # This gives us the time-weighted average of operational AI chips
            average_compute = h100_years / time_duration
            average_compute_at_threshold.append(average_compute)

        average_compute_ccdfs[lr] = calculate_ccdf(average_compute_at_threshold)

    return average_compute_ccdfs


def extract_project_time_to_detection_ccdfs(simulation_results, years, agreement_year, likelihood_ratios):
    """Calculate CCDFs for time to detection (length of slowdown) at different detection thresholds.

    Uses all simulations. Simulations where detection never happens get a large value (1000 years)
    to represent "never detected", which can be filtered and displayed separately in the frontend.
    """
    time_to_detection_ccdfs = {}

    # Large value to represent "never detected"
    NEVER_DETECTED_VALUE = 1000

    for lr in likelihood_ratios:
        time_at_threshold = []

        for black_projects in simulation_results:
            black_project = black_projects["prc_black_project"]

            # Use the covert project's years list
            sim_years = black_project.years

            detection_year = calculate_project_detection_year(black_project, sim_years, lr)
            if detection_year is None:
                # Detection never happened - use large value
                time_at_threshold.append(NEVER_DETECTED_VALUE)
            else:
                # Time to detection is detection year minus agreement year
                time_to_detection = detection_year - agreement_year
                time_at_threshold.append(time_to_detection)

        time_to_detection_ccdfs[lr] = calculate_ccdf(time_at_threshold)

    return time_to_detection_ccdfs


def calculate_datacenter_detection_year(black_datacenters, years, agreement_year, threshold):
    """Calculate the year when datacenter is detected based on likelihood ratio threshold.

    Detection occurs the first year the datacenter LR exceeds the threshold.
    """
    if black_datacenters is None:
        return None

    for year in years:
        relative_year = year - agreement_year
        year_lr = black_datacenters.cumulative_lr_from_concealed_datacenters(relative_year)

        if year_lr >= threshold:
            return year

    return None


def extract_datacenter_capacity_at_detection(simulation_results, years, agreement_year, threshold):
    """Extract total datacenter capacity (GW) at detection time (concealed + unconcealed)."""
    capacity_at_detection = []

    for black_projects in simulation_results:
        black_datacenters = black_projects["prc_black_project"].black_datacenters

        # Use simulation years to calculate detection
        sim_years = black_projects["prc_black_project"].years

        detection_year = calculate_datacenter_detection_year(black_datacenters, sim_years, agreement_year, threshold)

        if detection_year is not None:
            # Get capacity at detection year
            relative_year = detection_year - agreement_year
            capacity_gw = black_datacenters.get_covert_GW_capacity_total(relative_year)
        else:
            # Never detected - use final year
            final_year = years[-1]
            relative_year = final_year - agreement_year
            capacity_gw = black_datacenters.get_covert_GW_capacity_total(relative_year)

        capacity_at_detection.append(capacity_gw)

    return capacity_at_detection


def extract_datacenter_detection_statistics(simulation_results, years, agreement_year, detection_thresholds):
    """Extract datacenter capacity CCDFs for multiple detection thresholds."""
    capacity_ccdfs = {}

    for threshold in detection_thresholds:
        capacity_at_detection = extract_datacenter_capacity_at_detection(
            simulation_results, years, agreement_year, threshold
        )
        capacity_ccdfs[threshold] = calculate_ccdf(capacity_at_detection)

    return capacity_ccdfs


def extract_individual_datacenter_detection_data(simulation_results, years, agreement_year, threshold):
    """Extract individual datacenter detection data for dashboard display.

    Returns total capacity (concealed + unconcealed) at detection and years operational before detection.
    """
    individual_capacity = []
    individual_time = []

    for black_projects in simulation_results:
        black_datacenters = black_projects["prc_black_project"].black_datacenters

        # Use simulation years to calculate detection
        sim_years = black_projects["prc_black_project"].years

        detection_year = calculate_datacenter_detection_year(black_datacenters, sim_years, agreement_year, threshold)

        if detection_year is not None:
            # Get capacity at detection year
            relative_year = detection_year - agreement_year
            capacity_gw = black_datacenters.get_covert_GW_capacity_total(relative_year)
        else:
            # Never detected - use final year
            final_year = years[-1]
            relative_year = final_year - agreement_year
            capacity_gw = black_datacenters.get_covert_GW_capacity_total(relative_year)
            detection_year = final_year

        # Calculate years operational before detection
        # Datacenters start being built at agreement_year
        years_operational = detection_year - agreement_year

        individual_capacity.append(capacity_gw)
        individual_time.append(years_operational)

    return individual_capacity, individual_time


def extract_individual_project_detection_data(simulation_results, agreement_year, threshold, app_params):
    """Extract individual project detection data for dashboard display."""
    individual_h100e = []
    individual_energy = []
    individual_time = []
    individual_h100_years = []

    for black_projects in simulation_results:
        black_project = black_projects["prc_black_project"]
        sim_years = black_project.years

        detection_year = calculate_project_detection_year(black_project, sim_years, threshold)

        if detection_year is not None:
            operational_compute = black_project.operational_black_project(detection_year)
        else:
            final_year = max(sim_years)
            operational_compute = black_project.operational_black_project(final_year)
            detection_year = final_year

        operational_h100e = operational_compute.total_h100e_tpp()

        energy_gw = (operational_h100e * H100_TPP_PER_CHIP * H100_WATTS_PER_TPP /
                    app_params.black_project_parameters.exogenous_trends.energy_efficiency_of_prc_stock_relative_to_state_of_the_art / 1e9)

        time_operational = detection_year - agreement_year
        h100_years = calculate_h100_years_until_detection(black_project, sim_years, detection_year)

        individual_h100e.append(operational_h100e)
        individual_energy.append(energy_gw)
        individual_time.append(time_operational)
        individual_h100_years.append(h100_years)

    return individual_h100e, individual_energy, individual_time, individual_h100_years


def extract_ai_rd_reduction_ccdfs(simulation_results, years, agreement_year, likelihood_ratios, app_params):
    """Calculate CCDFs for largest company and PRC AI R&D compute reduction at different detection thresholds.

    For each simulation and detection threshold:
    1. Calculate detection year based on likelihood ratio
    2. Calculate total covert project H100-years from agreement to detection
    3. Calculate total largest company and PRC AI R&D H100-years (no slowdown counterfactual) for same period
    4. Return ratios: (AI R&D H100-years) / (covert project H100-years)

    This represents how much larger AI R&D compute would be without the slowdown.
    """
    largest_company_reduction_ccdfs = {}
    prc_reduction_ccdfs = {}

    slowdown_params = app_params.black_project_parameters.slowdown_counterfactual_parameters

    # PRC parameters
    exogenous_trends = app_params.black_project_parameters.exogenous_trends
    prc_compute_2025 = exogenous_trends.total_prc_compute_stock_in_2025
    prc_growth_rate = exogenous_trends.annual_growth_rate_of_prc_compute_stock_p50
    prc_ai_rd_fraction = slowdown_params.fraction_of_prc_compute_spent_on_ai_rd_before_slowdown

    # Largest AI project
    largest_ai_project = LargestAIProject(exogenous_trends)

    for lr in likelihood_ratios:
        largest_company_reduction_ratios = []
        prc_reduction_ratios = []

        for black_projects in simulation_results:
            black_project = black_projects["prc_black_project"]
            sim_years = black_project.years

            # Calculate detection year for this simulation
            detection_year = calculate_project_detection_year(black_project, sim_years, lr)
            if detection_year is None:
                detection_year = max(sim_years)

            # Calculate total covert project H100-years from agreement to detection
            # Using the h100_years_to_date method which properly integrates operational compute over time
            covert_h100_years = black_project.h100_years_to_date(detection_year, sim_years)

            # Calculate total largest AI project H100-years (no slowdown counterfactual)
            # Need to integrate over time: sum of (H100e * time_step)
            largest_company_h100_years = 0
            years_in_range = [y for y in sim_years if agreement_year <= y <= detection_year]
            for i in range(len(years_in_range) - 1):
                year = years_in_range[i]
                next_year = years_in_range[i + 1]
                time_step = next_year - year
                # Get compute stock at this year
                largest_company_h100_years += largest_ai_project.get_compute_stock(year) * time_step

            # Calculate total PRC AI R&D H100-years (no slowdown counterfactual)
            prc_h100_years = 0
            for i in range(len(years_in_range) - 1):
                year = years_in_range[i]
                next_year = years_in_range[i + 1]
                time_step = next_year - year
                years_since_2025 = year - 2025
                prc_compute = prc_compute_2025 * (prc_growth_rate ** years_since_2025)
                # Apply fraction for PRC AI R&D and multiply by time step
                prc_h100_years += prc_compute * prc_ai_rd_fraction * time_step

            # Calculate reduction ratios (H100-years / H100-years)
            # If no covert compute, use a large value to represent "infinite" reduction
            if covert_h100_years <= 0:
                large_reduction = 1e12
                largest_company_reduction_ratios.append(large_reduction)
                prc_reduction_ratios.append(large_reduction)
            else:
                largest_company_ratio = largest_company_h100_years / covert_h100_years
                largest_company_reduction_ratios.append(largest_company_ratio)

                prc_ratio = prc_h100_years / covert_h100_years
                prc_reduction_ratios.append(prc_ratio)

        # Calculate CCDFs for this threshold
        largest_company_reduction_ccdfs[lr] = calculate_ccdf(largest_company_reduction_ratios)
        prc_reduction_ccdfs[lr] = calculate_ccdf(prc_reduction_ratios)

    return {"largest_company": largest_company_reduction_ccdfs, "prc": prc_reduction_ccdfs}


def extract_chip_production_reduction_ccdfs(simulation_results, years, agreement_year, likelihood_ratios, app_params):
    """Calculate CCDFs for global, largest company and PRC AI chip production reduction at different detection thresholds.

    For each simulation and detection threshold:
    1. Calculate detection year based on likelihood ratio
    2. Calculate total covert fab chip production from agreement to detection (PRC with slowdown)
    3. Calculate total global chip production (no slowdown counterfactual) for same period
    4. Calculate total largest company chip production (no slowdown counterfactual) for same period
    5. Calculate total PRC chip production (no slowdown counterfactual) for same period
    6. Return ratios: (counterfactual production) / (covert fab production)

    Chip production is calculated as the change in stock from start to end of period.
    For exponentially growing stock S(t) = S_0 * g^t, production = S(end) - S(start).
    Global production uses data from global_compute_production.csv.
    """
    global_production_ccdfs = {}
    largest_company_production_ccdfs = {}
    prc_production_ccdfs = {}

    # PRC parameters
    exogenous_trends = app_params.black_project_parameters.exogenous_trends
    prc_compute_2025 = exogenous_trends.total_prc_compute_stock_in_2025
    prc_growth_rate = exogenous_trends.annual_growth_rate_of_prc_compute_stock_p50

    # Largest AI project
    largest_ai_project = LargestAIProject(exogenous_trends)

    for lr in likelihood_ratios:
        global_production_ratios = []
        largest_company_production_ratios = []
        prc_production_ratios = []

        for black_projects in simulation_results:
            black_project = black_projects["prc_black_project"]
            black_fab = black_project.black_fab
            sim_years = black_project.years

            # Calculate detection year for this simulation
            detection_year = calculate_project_detection_year(black_project, sim_years, lr)
            if detection_year is None:
                detection_year = max(sim_years)

            # Get covert fab chip production from agreement to detection
            black_fab_production = 0.0
            if black_fab is not None:
                cumulative_by_year = black_fab.get_cumulative_compute_production_over_time()
                # Get production at detection year (cumulative) minus production at agreement year
                prod_at_detection = cumulative_by_year.get(detection_year, 0.0)
                prod_at_agreement = cumulative_by_year.get(agreement_year, 0.0)
                # If agreement year not in dict, find closest year before
                if agreement_year not in cumulative_by_year:
                    earlier_years = [y for y in cumulative_by_year.keys() if y < agreement_year]
                    if earlier_years:
                        prod_at_agreement = cumulative_by_year.get(max(earlier_years), 0.0)
                    else:
                        prod_at_agreement = 0.0
                # If detection year not in dict, find closest year before
                if detection_year not in cumulative_by_year:
                    earlier_years = [y for y in cumulative_by_year.keys() if y <= detection_year]
                    if earlier_years:
                        prod_at_detection = cumulative_by_year.get(max(earlier_years), 0.0)
                    else:
                        prod_at_detection = 0.0
                black_fab_production = prod_at_detection - prod_at_agreement

            # Calculate global chip production from agreement to detection (no slowdown)
            # Uses data from global_compute_production.csv
            global_production = get_global_compute_production_between_years(agreement_year, detection_year)

            # Calculate largest AI project chip production from agreement to detection (no slowdown)
            # Production = Stock(detection) - Stock(agreement)
            largest_company_stock_at_agreement = largest_ai_project.get_compute_stock(agreement_year)
            largest_company_stock_at_detection = largest_ai_project.get_compute_stock(detection_year)
            largest_company_production = largest_company_stock_at_detection - largest_company_stock_at_agreement

            # Calculate PRC chip production from agreement to detection (no slowdown)
            years_since_2025_agreement = agreement_year - 2025
            years_since_2025_detection = detection_year - 2025
            prc_stock_at_agreement = prc_compute_2025 * (prc_growth_rate ** years_since_2025_agreement)
            prc_stock_at_detection = prc_compute_2025 * (prc_growth_rate ** years_since_2025_detection)
            prc_production = prc_stock_at_detection - prc_stock_at_agreement

            # Calculate reduction ratios
            # If no covert fab production, use a large value to represent "infinite" reduction
            if black_fab_production <= 0:
                # Use a very large value (1e12) to represent infinite reduction
                # This will appear on the far right of the CCDF
                large_reduction = 1e12
                global_production_ratios.append(large_reduction)
                largest_company_production_ratios.append(large_reduction)
                prc_production_ratios.append(large_reduction)
            else:
                if global_production > 0:
                    global_ratio = global_production / black_fab_production
                    global_production_ratios.append(global_ratio)
                else:
                    global_production_ratios.append(0.0)

                largest_company_ratio = largest_company_production / black_fab_production
                largest_company_production_ratios.append(largest_company_ratio)

                prc_ratio = prc_production / black_fab_production
                prc_production_ratios.append(prc_ratio)

        # Calculate CCDFs for this threshold
        global_production_ccdfs[lr] = calculate_ccdf(global_production_ratios)
        largest_company_production_ccdfs[lr] = calculate_ccdf(largest_company_production_ratios)
        prc_production_ccdfs[lr] = calculate_ccdf(prc_production_ratios)

    return {"global": global_production_ccdfs, "largest_company": largest_company_production_ccdfs, "prc": prc_production_ccdfs}


def extract_takeoff_slowdown_trajectories(simulation_results, years, agreement_year, app_params):
    """Extract compute trajectories for AI takeoff slowdown analysis.

    Returns two time series of STOCK (instantaneous capacity):
    1. Largest company AI R&D compute stock (no slowdown counterfactual)
    2. Median covert project operational compute stock (with slowdown)

    Both in units of H100-equivalents (stock, not cumulative flow).
    The takeoff model expects stock values at each time point.
    Also predicts milestone times using the takeoff model.
    """
    # Extend years range for trajectory prediction (need to go beyond simulation end)
    extended_years = list(range(int(years[0]), int(years[-1]) + 16))  # Extend 15 years beyond

    # Largest AI project
    exogenous_trends = app_params.black_project_parameters.exogenous_trends
    largest_ai_project = LargestAIProject(exogenous_trends)

    # 1. Largest AI project compute STOCK trajectory (no slowdown)
    largest_company_ai_rd_stock = largest_ai_project.get_compute_trajectory([float(y) for y in extended_years])

    # 2. Covert project operational compute STOCK (median across simulations)
    # This is the instantaneous stock of operational H100e at each year
    # For years beyond simulation, assume constant (conservative estimate)
    operational_by_sim = []
    for black_projects in simulation_results:
        operational_series = []
        last_value = 0
        for year in extended_years:
            if year <= years[-1]:
                # Within simulation range: get operational stock at this year
                operational_compute = black_projects["prc_black_project"].operational_black_project(year)
                value = operational_compute.total_h100e_tpp()
                operational_series.append(value)
                last_value = value
            else:
                # Beyond simulation range: use last known value
                operational_series.append(last_value)
        operational_by_sim.append(operational_series)

    # Calculate median covert compute stock
    operational_array = np.array(operational_by_sim)
    covert_median_stock = np.median(operational_array, axis=0)

    # The values are already in H100e TPP (stock), which the takeoff model can use directly
    # No need for utilization adjustment - these are capacity values

    # Predict milestones using takeoff model (optional - only if available)
    milestones_global = None
    milestones_covert = None
    trajectory_global = None
    trajectory_covert = None
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

        # Predict for global scenario
        # The takeoff model expects H100-years as input, which represents compute stock
        # Our H100e TPP values are already stock values and can be used directly
        years_array = np.array(extended_years, dtype=float)
        global_array = np.array(largest_company_ai_rd_stock, dtype=float)
        covert_array = np.array(covert_median_stock, dtype=float)

        print(f"Predicting trajectories: years {years_array[0]} to {years_array[-1]}")
        print(f"Largest company stock range: {global_array[0]:.2e} to {global_array[-1]:.2e} H100e")
        print(f"Covert stock range: {covert_array[0]:.2e} to {covert_array[-1]:.2e} H100e")

        # Use TrajectoryPredictor to get full trajectory including time series
        predictor_global = TrajectoryPredictor()

        # Split total compute into inference and experiment (15% inference, 85% experiment)
        inference_fraction = 0.15
        global_inference = global_array * inference_fraction
        global_experiment = global_array * (1 - inference_fraction)

        milestones_global_dict = predictor_global.predict_from_time_series(
            time=years_array,
            inference_compute=global_inference,
            experiment_compute=global_experiment
        )

        # Get full trajectory for global scenario
        trajectory_global = predictor_global.get_full_trajectory()

        # Same for covert scenario
        predictor_covert = TrajectoryPredictor()
        covert_inference = covert_array * inference_fraction
        covert_experiment = covert_array * (1 - inference_fraction)

        milestones_covert_dict = predictor_covert.predict_from_time_series(
            time=years_array,
            inference_compute=covert_inference,
            experiment_compute=covert_experiment
        )

        # Get full trajectory for covert scenario
        trajectory_covert = predictor_covert.get_full_trajectory()

        # Extract key milestone times with progress_multiplier
        milestones_global = {
            name: {
                'time': float(info.time),
                'progress_level': float(info.progress_level),
                'progress_multiplier': float(info.progress_multiplier)
            }
            for name, info in milestones_global_dict.items()
        }

        milestones_covert = {
            name: {
                'time': float(info.time),
                'progress_level': float(info.progress_level),
                'progress_multiplier': float(info.progress_multiplier)
            }
            for name, info in milestones_covert_dict.items()
        }

        print(f"Successfully predicted {len(milestones_global)} milestones for each scenario")

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
    covert_ai_speedup = None
    trajectory_times = None

    if trajectory_global and 'times' in trajectory_global and 'ai_sw_progress_mult_ref_present_day' in trajectory_global:
        # Convert to list if needed
        times = trajectory_global['times']
        trajectory_times = times.tolist() if hasattr(times, 'tolist') else list(times)

        speedup = trajectory_global['ai_sw_progress_mult_ref_present_day']
        global_ai_speedup = speedup.tolist() if hasattr(speedup, 'tolist') else list(speedup)

    if trajectory_covert and 'ai_sw_progress_mult_ref_present_day' in trajectory_covert:
        speedup = trajectory_covert['ai_sw_progress_mult_ref_present_day']
        covert_ai_speedup = speedup.tolist() if hasattr(speedup, 'tolist') else list(speedup)

    return {
        'years': extended_years,
        'largest_company_ai_rd_stock': largest_company_ai_rd_stock,
        'covert_stock': covert_median_stock.tolist(),
        'milestones_global': milestones_global,
        'milestones_covert': milestones_covert,
        'trajectory_times': trajectory_times,
        'global_ai_speedup': global_ai_speedup,
        'covert_ai_speedup': covert_ai_speedup
    }


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
    p_project_exists = app_params.black_project_parameters.p_project_exists
    if not model.simulation_results:
        return {"error": "No simulation results"}

    agreement_year = app_params.simulation_settings.agreement_start_year
    years = get_years_from_simulation(model.simulation_results)
    years_array = np.array(years)

    # Track which simulations have fabs built
    fab_built_in_sim = [
        cp["prc_black_project"].black_fab is not None
        for cp in model.simulation_results
    ]

    # Extract time series data (needed for calculations)
    h100e_by_sim = extract_fab_production_over_time(model.simulation_results, years)
    survival_rate_by_sim = extract_survival_rates_over_time(model.simulation_results, years)
    black_project_by_sim = extract_black_project_over_time(model.simulation_results, years)
    operational_black_project_by_sim = extract_operational_black_project_over_time(model.simulation_results, years)
    datacenter_capacity_by_sim = extract_datacenter_capacity_over_time(model.simulation_results, years, agreement_year)
    lr_datacenters_by_sim = extract_datacenter_lr_over_time(model.simulation_results, years, agreement_year)
    h100_years_by_sim = extract_h100_years_over_time(model.simulation_results, years, agreement_year)
    cumulative_lr_by_sim = extract_cumulative_lr_over_time(model.simulation_results, years)
    posterior_prob_by_sim = extract_posterior_prob_over_time(cumulative_lr_by_sim, p_project_exists)
    lr_initial_by_sim, lr_sme_by_sim, lr_other_by_sim = extract_project_lr_components_over_time(model.simulation_results, years)
    lr_prc_accounting_by_sim, lr_sme_inventory_by_sim = extract_detailed_lr_components_over_time(model.simulation_results, years)

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
    # For datacenter capacity, only use 1x threshold since detection is binary
    datacenter_capacity_ccdfs = extract_datacenter_detection_statistics(
        model.simulation_results, years, agreement_year, [1]
    )
    # Use LR=5 as the dashboard threshold (matches default in frontend)
    dashboard_lr_threshold = 5
    individual_fab_h100e, individual_fab_time, individual_fab_process_nodes, individual_fab_energy = \
        extract_individual_fab_detection_data(model.simulation_results, years, dashboard_lr_threshold, app_params)
    individual_project_h100e, individual_project_energy, individual_project_time, individual_project_h100_years = \
        extract_individual_project_detection_data(model.simulation_results, agreement_year, dashboard_lr_threshold, app_params)
    individual_datacenter_capacity, individual_datacenter_time = \
        extract_individual_datacenter_detection_data(model.simulation_results, years, agreement_year, dashboard_lr_threshold)

    # Extract initial stock data
    diversion_proportion = app_params.black_project_properties.proportion_of_initial_compute_stock_to_divert
    initial_stock_data = extract_initial_stock_data(model.simulation_results, likelihood_ratios, diversion_proportion)

    # Extract PRC compute over time (from 2025 to agreement start year)
    exogenous_trends = app_params.black_project_parameters.exogenous_trends
    prc_compute_years = list(range(2025, int(agreement_year) + 1))
    prc_compute_over_time_by_sim = extract_prc_compute_over_time(prc_compute_years, exogenous_trends)

    # Calculate domestic production (proportion of total PRC compute with interpolation)
    prc_domestic_compute_by_sim = []
    for sim_data in prc_compute_over_time_by_sim:
        domestic_compute_for_sim = []
        for year, compute in zip(prc_compute_years, sim_data):
            proportion_domestic = interpolate_domestic_proportion(year, exogenous_trends)
            domestic_compute_for_sim.append(compute * proportion_domestic)
        prc_domestic_compute_by_sim.append(domestic_compute_for_sim)

    # Calculate largest AI project compute over time
    largest_ai_project = LargestAIProject(exogenous_trends)
    largest_company_compute_over_time = extract_largest_company_compute_over_time(prc_compute_years, largest_ai_project)

    # Calculate interpolated domestic proportion for each year
    proportion_domestic_by_year = [
        interpolate_domestic_proportion(year, exogenous_trends)
        for year in prc_compute_years
    ]

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
        "researcher_headcount": app_params.black_project_properties.researcher_headcount,

        # =====================================================================
        # DARK COMPUTE MODEL (main page overview)
        # =====================================================================
        "black_project_model": {
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
            "likelihood_ratios": likelihood_ratios,

            # -------------------------------------------------------------------
            # Average Covert Compute Operated CCDF Plot
            # -------------------------------------------------------------------
            "average_covert_compute_ccdf": extract_average_covert_compute_ccdfs(
                model.simulation_results, years, agreement_year, likelihood_ratios
            ),

            # -------------------------------------------------------------------
            # AI R&D Reduction CCDF Plot
            # -------------------------------------------------------------------
            "ai_rd_reduction_ccdf": extract_ai_rd_reduction_ccdfs(
                model.simulation_results, years, agreement_year, likelihood_ratios, app_params
            ),

            # -------------------------------------------------------------------
            # Chip Production Reduction CCDF Plot
            # -------------------------------------------------------------------
            "chip_production_reduction_ccdf": extract_chip_production_reduction_ccdfs(
                model.simulation_results, years, agreement_year, likelihood_ratios, app_params
            ),

            # -------------------------------------------------------------------
            # Time to Detection CCDF Plot
            # -------------------------------------------------------------------
            "time_to_detection_ccdf": extract_project_time_to_detection_ccdfs(
                model.simulation_results, years, agreement_year, likelihood_ratios
            ),

            # -------------------------------------------------------------------
            # H100-Years and Evidence Over Time Plot
            # -------------------------------------------------------------------
            "h100_years": fmt_pct(h100_years_by_sim),
            "cumulative_lr": fmt_pct(cumulative_lr_by_sim),

            # -------------------------------------------------------------------
            # Dark Compute Stock Breakdown
            # -------------------------------------------------------------------
            "initial_black_project": fmt_pct(black_project_by_sim),  # Keep in raw H100e units with k/M formatting
            "black_fab_flow": fmt_pct(filter_to_with_fab(h100e_by_sim, fab_built_in_sim)),  # Filtered - for Covert fab section plots
            "black_fab_flow_all_sims": fmt_pct(h100e_by_sim),  # All simulations - for Dark Compute Stock Breakdown
            "survival_rate": fmt_pct(survival_rate_by_sim),
            "total_black_project": fmt_pct(black_project_by_sim),  # Keep in raw H100e units with k/M formatting

            # -------------------------------------------------------------------
            # Data Center Capacity Breakdown
            # -------------------------------------------------------------------
            "datacenter_capacity": fmt_pct(datacenter_capacity_by_sim),
            "black_project_energy": energy_by_source_array.tolist(),
            "energy_source_labels": source_labels,
            "operational_black_project": fmt_pct(operational_black_project_by_sim),  # Keep in raw H100e units with k/M formatting

            # -------------------------------------------------------------------
            # Strength of Evidence Breakdown
            # -------------------------------------------------------------------
            "lr_initial_stock": fmt_pct(lr_initial_by_sim),
            "lr_diverted_sme": fmt_pct(lr_sme_by_sim),
            "lr_other_intel": fmt_pct(lr_other_by_sim),
            "posterior_prob_project": fmt_pct(posterior_prob_by_sim),

            # Individual LR components for detailed breakdown
            "lr_prc_accounting": fmt_pct(lr_prc_accounting_by_sim),
            "lr_sme_inventory": fmt_pct(lr_sme_inventory_by_sim),

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
        "initial_black_project": {
            "years": years_array.tolist(),
            # Covert fab production flow (cumulative over time)
            "h100e": fmt_pct(filter_to_with_fab(h100e_by_sim, fab_built_in_sim), convert_to_thousands),
            # Chip survival rate over time
            "survival_rate": fmt_pct(survival_rate_by_sim),
            # Total dark compute (surviving, not capacity-limited)
            "black_project": fmt_pct(black_project_by_sim, convert_to_thousands),
        },

        # =====================================================================
        # COVERT DATA CENTERS
        # =====================================================================
        "black_datacenters": {
            "years": years_array.tolist(),
            # Datacenter capacity (GW)
            "datacenter_capacity": fmt_pct(datacenter_capacity_by_sim),
            # Energy consumption by source (stacked area plot)
            "energy_by_source": energy_by_source_array.tolist(),
            "source_labels": source_labels,
            # Operational dark compute (capacity-limited)
            "operational_black_project": fmt_pct(operational_black_project_by_sim, convert_to_thousands),
            # Datacenter detection
            "lr_datacenters": fmt_pct(lr_datacenters_by_sim),
            "datacenter_detection_prob": (np.mean(np.array(lr_datacenters_by_sim) >= 5.0, axis=0)).tolist(),
            # Datacenter capacity CCDFs at detection
            "capacity_ccdfs": datacenter_capacity_ccdfs,
            "likelihood_ratios": likelihood_ratios,
            # Dashboard - Individual simulation data
            "individual_capacity_before_detection": individual_datacenter_capacity,
            "individual_time_before_detection": individual_datacenter_time,
            # PRC datacenter capacity trajectory (from 2025 to agreement year, using medians)
            **calculate_prc_datacenter_capacity_trajectory(app_params),
            # Fraction diverted to covert project
            "fraction_diverted": app_params.black_project_properties.fraction_of_datacenter_capacity_not_built_for_concealment_diverted_to_black_project_at_agreement_start,
        },

        # =====================================================================
        # COVERT FAB
        # =====================================================================
        "black_fab": {
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
            "watts_per_tpp_curve": PRCBlackFab.watts_per_tpp_relative_to_H100(),
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
        "initial_stock": {
            **initial_stock_data,
            "prc_compute_over_time": fmt_pct(prc_compute_over_time_by_sim),
            "prc_domestic_compute_over_time": fmt_pct(prc_domestic_compute_by_sim),
            "largest_company_compute_over_time": largest_company_compute_over_time,
            "prc_compute_years": prc_compute_years,
            "proportion_domestic_by_year": proportion_domestic_by_year,
            "years": years_array.tolist(),
            "state_of_the_art_energy_efficiency_relative_to_h100": largest_ai_project.get_energy_efficiency_relative_to_h100(agreement_year),
        },
    }

    # Convert numpy types to native Python types for JSON serialization
    converted_results = convert_numpy_types(results)

    # Debug: Print covert fab CCDF data
    print("\n" + "="*80)
    print("COVERT FAB COMPUTE CCDFs:")
    print(f"Likelihood ratios: {converted_results['black_fab']['likelihood_ratios']}")
    print(f"Compute CCDFs keys: {list(converted_results['black_fab']['compute_ccdfs'].keys())}")
    for lr, ccdf_data in converted_results['black_fab']['compute_ccdfs'].items():
        print(f"\nLR={lr}: {len(ccdf_data)} data points")
        if ccdf_data:
            print(f"  First 3 points: {ccdf_data[:3]}")
            print(f"  Last 3 points: {ccdf_data[-3:]}")
    print("="*80 + "\n")

    # Debug: Print project H100-years CCDF data
    print("\n" + "="*80)
    print("PROJECT H100-YEARS CCDFs:")
    print(f"Likelihood ratios: {converted_results['black_project_model']['likelihood_ratios']}")
    print(f"H100-years CCDFs keys: {list(converted_results['black_project_model']['h100_years_ccdf'].keys())}")
    for lr, ccdf_data in converted_results['black_project_model']['h100_years_ccdf'].items():
        print(f"\nLR={lr}: {len(ccdf_data)} data points")
        if ccdf_data:
            print(f"  First 3 points: {ccdf_data[:3]}")
            print(f"  Last 3 points: {ccdf_data[-3:]}")
    print("="*80 + "\n")

    return converted_results
