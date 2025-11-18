from dataclasses import dataclass
from typing import Optional
from enum import Enum

class ProcessNode(Enum):
    nm130 = "130nm"
    nm28 = "28nm"
    nm14 = "14nm"
    nm7 = "7nm"

    def to_nm(self) -> int:
        """Extract the numeric nanometer value from the enum."""
        return int(self.value.replace("nm", ""))

@dataclass
class SimulationSettings:
    start_year : int = 2030
    end_year : int = 2037
    time_step_years : float = 0.1
    num_simulations : int = 60

@dataclass
class CovertProjectStrategy:
    run_a_covert_project : bool = True

    # Initial compute stock
    proportion_of_initial_compute_stock_to_divert : Optional[float] = 0.05

    # Data centers
    GW_per_initial_datacenter : float = 5
    number_of_initial_datacenters : float = 0.1
    GW_per_year_of_concealed_datacenters : float = 1

    # Covert fab
    build_a_covert_fab : bool = True
    covert_fab_operating_labor : Optional[int] = 550
    covert_fab_construction_labor : Optional[int] = 250
    covert_fab_process_node : Optional[ProcessNode] = "best_indigenous"
    covert_fab_proportion_of_prc_lithography_scanners_devoted : Optional[float] = 0.1

class InitialPRCDarkComputeParameters():

    # H100 power consumption
    h100_power_watts = 700  # Total power consumption of NVIDIA H100 GPU

    # PRC compute stock
    total_prc_compute_stock_in_2025 = 1e6
    energy_efficiency_relative_to_h100 = 0.5
    annual_growth_rate_of_prc_compute_stock = 2.4
    relative_sigma_of_prc_compute_stock = 0.1

    us_intelligence_median_error_in_estimate_of_prc_compute_stock = 0.07

    # Global compute stock parameters
    total_global_compute_in_2025 = 1e7
    annual_growth_rate_of_global_compute = 2.4
    relative_sigma_of_global_compute = 0.1

    median_unreported_compute_owned_by_non_prc_actors = 1e6
    relative_sigma_unreported_compute_owned_by_non_prc_actors = 0.5

class SurvivalRateParameters():

    # Survival rate parameters
    initial_hazard_rate_p50 = 0.01
    increase_of_hazard_rate_per_year_p50 = 0.0035
    hazard_rate_p25_relative_to_p50 = 0.1
    hazard_rate_p75_relative_to_p50 = 6

@dataclass
class CovertDatacenterParameters:
    # -- Energy capacity --
    max_proportion_of_PRC_energy_consumption = 0.05
    total_GW_of_PRC_energy_consumption = 1100

    # -- Detection evidence --
    construction_labor_per_MW_per_year = 6.3
    relative_sigma_construction_labor_per_MW_per_year = 0.4

    operating_labor_per_MW = 1
    relative_sigma_operating_labor_per_MW = 0.4

    # Now replaced with corresponding parameters in "CovertProject"
    # mean_detection_time_of_covert_site_for_100_workers = 6.95
    # mean_detection_time_of_covert_site_for_1000_workers = 3.42
    # variance_of_detection_time_given_num_workers = 3.880

@dataclass
class CovertFabParameters:

    # Odds of covert project
    median_absolute_relative_error_of_us_intelligence_estimate_of_prc_sme_stock = 0.07
    # Now replaced with corresponding parameters in "CovertProject"
    # mean_detection_time_for_100_workers = 6.95
    # mean_detection_time_for_1000_workers = 3.42
    # variance_of_detection_time_given_num_workers = 3.880
    wafers_per_month_per_worker = 24.64
    labor_productivity_relative_sigma = 0.62
    wafers_per_month_per_lithography_scanner = 1000
    scanner_productivity_relative_sigma = 0.20
    Probability_of_90p_PRC_localization_at_node = {
        ProcessNode.nm130: [(2025, 0.80), (2031, 0.80)],
        ProcessNode.nm28: [(2025, 0.0), (2031, 0.25)],
        ProcessNode.nm14: [(2025, 0.0), (2031, 0.10)],
        ProcessNode.nm7: [(2025, 0.0), (2031, 0.06)]
    }

    # Compute Production Rate
    construction_time_for_5k_wafers_per_month = 1.40
    construction_time_for_100k_wafers_per_month = 2.41
    construction_time_relative_sigma = 0.35
    construction_workers_per_1000_wafers_per_month = 14.1
    h100_sized_chips_per_wafer = 28
    transistor_density_scaling_exponent = 1.49
    architecture_efficiency_improvement_per_year = 1.23
    prc_additional_lithography_scanners_produced_per_year = 16.0
    prc_lithography_scanners_produced_in_first_year = 20.0
    prc_scanner_production_relative_sigma = 0.30

    # Energy requirements
    watts_per_tpp_vs_transistor_density_exponent_before_dennard_scaling_ended = -2.00
    watts_per_tpp_vs_transistor_density_exponent_after_dennard_scaling_ended = -0.91
    transistor_density_at_end_of_dennard_scaling_m_per_mm2 = 1.98

@dataclass
class CovertProjectParameters:
    mean_detection_time_for_100_workers = 6.95
    mean_detection_time_for_1000_workers = 3.42
    variance_of_detection_time_given_num_workers = 3.880
    initial_compute_stock_parameters = InitialPRCDarkComputeParameters()
    survival_rate_parameters = SurvivalRateParameters()
    datacenter_model_parameters = CovertDatacenterParameters()
    covert_fab_parameters = CovertFabParameters()

@dataclass
class Parameters:
    simulation_settings : SimulationSettings
    covert_project_strategy : CovertProjectStrategy
    covert_project_parameters : CovertProjectParameters

