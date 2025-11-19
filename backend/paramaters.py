from dataclasses import dataclass, field
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

class ProcessNodeStrategy(Enum):
    """Strategy for selecting process node for covert fab"""
    BEST_INDIGENOUS = "best_indigenous"
    BEST_INDIGENOUS_GTE_28NM = "best_indigenous_gte_28nm"
    BEST_INDIGENOUS_GTE_14NM = "best_indigenous_gte_14nm"
    BEST_INDIGENOUS_GTE_7NM = "best_indigenous_gte_7nm"

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
    datacenter_construction_labor : int = 500

    # Covert fab
    build_a_covert_fab : bool = True
    covert_fab_operating_labor : Optional[int] = 550
    covert_fab_construction_labor : Optional[int] = 250
    covert_fab_process_node : Optional[ProcessNodeStrategy] = ProcessNodeStrategy.BEST_INDIGENOUS_GTE_28NM
    covert_fab_proportion_of_prc_lithography_scanners_devoted : Optional[float] = 0.1

@dataclass
class InitialPRCDarkComputeParameters:

    # H100 power consumption
    h100_power_watts: float = 700  # Total power consumption of NVIDIA H100 GPU

    # PRC compute stock
    total_prc_compute_stock_in_2025: float = 1e6
    energy_efficiency_relative_to_h100: float = 0.5
    annual_growth_rate_of_prc_compute_stock: float = 2.4
    relative_sigma_of_prc_compute_stock: float = 0.1

    us_intelligence_median_error_in_estimate_of_prc_compute_stock: float = 0.07

    # Global compute stock parameters
    total_global_compute_in_2025: float = 1e7
    annual_growth_rate_of_global_compute: float = 2.4
    relative_sigma_of_global_compute: float = 0.1

    median_unreported_compute_owned_by_non_prc_actors: float = 1e6
    relative_sigma_unreported_compute_owned_by_non_prc_actors: float = 0.5

@dataclass
class SurvivalRateParameters:

    # Survival rate parameters
    initial_hazard_rate_p50: float = 0.01
    increase_of_hazard_rate_per_year_p50: float = 0.0035
    hazard_rate_p25_relative_to_p50: float = 0.1
    hazard_rate_p75_relative_to_p50: float = 6

@dataclass
class CovertDatacenterParameters:
    # -- Energy capacity --
    max_proportion_of_PRC_energy_consumption: float = 0.05
    total_GW_of_PRC_energy_consumption: float = 1100

    # -- Detection evidence --
    construction_labor_per_MW_per_year: float = 6.3
    relative_sigma_construction_labor_per_MW_per_year: float = 0.4

    operating_labor_per_MW: float = 1
    relative_sigma_operating_labor_per_MW: float = 0.4

    # Now replaced with corresponding parameters in "CovertProject"
    # mean_detection_time_of_covert_site_for_100_workers = 6.95
    # mean_detection_time_of_covert_site_for_1000_workers = 3.42
    # variance_of_detection_time_given_num_workers = 3.880

@dataclass
class CovertFabParameters:

    # Odds of covert project
    median_absolute_relative_error_of_us_intelligence_estimate_of_prc_sme_stock: float = 0.07
    # Now replaced with corresponding parameters in "CovertProject"
    # mean_detection_time_for_100_workers = 6.95
    # mean_detection_time_for_1000_workers = 3.42
    # variance_of_detection_time_given_num_workers = 3.880
    wafers_per_month_per_worker: float = 24.64
    labor_productivity_relative_sigma: float = 0.62
    wafers_per_month_per_lithography_scanner: float = 1000
    scanner_productivity_relative_sigma: float = 0.20

    # Localization probabilities (individual fields instead of dict for easier serialization)
    localization_130nm_2025: float = 0.80
    localization_130nm_2031: float = 0.80
    localization_28nm_2025: float = 0.0
    localization_28nm_2031: float = 0.25
    localization_14nm_2025: float = 0.0
    localization_14nm_2031: float = 0.10
    localization_7nm_2025: float = 0.0
    localization_7nm_2031: float = 0.06

    # Compute Production Rate
    construction_time_for_5k_wafers_per_month: float = 1.40
    construction_time_for_100k_wafers_per_month: float = 2.41
    construction_time_relative_sigma: float = 0.35
    construction_workers_per_1000_wafers_per_month: float = 14.1
    h100_sized_chips_per_wafer: float = 28
    transistor_density_scaling_exponent: float = 1.49
    architecture_efficiency_improvement_per_year: float = 1.23
    prc_additional_lithography_scanners_produced_per_year: float = 16.0
    prc_lithography_scanners_produced_in_first_year: float = 20.0
    prc_scanner_production_relative_sigma: float = 0.30

    # Energy requirements
    watts_per_tpp_vs_transistor_density_exponent_before_dennard_scaling_ended: float = -2.00
    watts_per_tpp_vs_transistor_density_exponent_after_dennard_scaling_ended: float = -0.91
    transistor_density_at_end_of_dennard_scaling_m_per_mm2: float = 1.98

    def get_probability_of_90p_prc_localization_at_node(self) -> dict:
        """Convert individual localization fields back to dict format for code that needs it."""
        return {
            ProcessNode.nm130: [(2025, self.localization_130nm_2025), (2031, self.localization_130nm_2031)],
            ProcessNode.nm28: [(2025, self.localization_28nm_2025), (2031, self.localization_28nm_2031)],
            ProcessNode.nm14: [(2025, self.localization_14nm_2025), (2031, self.localization_14nm_2031)],
            ProcessNode.nm7: [(2025, self.localization_7nm_2025), (2031, self.localization_7nm_2031)]
        }

@dataclass
class CovertProjectParameters:
    p_project_exists: float = 0.2
    mean_detection_time_for_100_workers: float = 6.95
    mean_detection_time_for_1000_workers: float = 3.42
    variance_of_detection_time_given_num_workers: float = 3.880
    initial_compute_stock_parameters: InitialPRCDarkComputeParameters = field(default_factory=InitialPRCDarkComputeParameters)
    survival_rate_parameters: SurvivalRateParameters = field(default_factory=SurvivalRateParameters)
    datacenter_model_parameters: CovertDatacenterParameters = field(default_factory=CovertDatacenterParameters)
    covert_fab_parameters: CovertFabParameters = field(default_factory=CovertFabParameters)

def _set_nested_attr(obj, path, value):
    """Set a nested attribute using dot notation: 'a.b.c' """
    parts = path.split('.')
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)

@dataclass
class Parameters:
    simulation_settings : SimulationSettings
    covert_project_strategy : CovertProjectStrategy
    covert_project_parameters : CovertProjectParameters

    def update_from_dict(self, data: dict):
        """
        Update parameters from request data using dot notation.
        Field names with dots (e.g., 'simulation_settings.start_year') are automatically
        converted to nested attribute access.
        Returns True if survival rate parameters changed (requires metalog cache clear).
        """
        for field_name, value in data.items():
            # Skip special cases handled below
            if field_name.startswith('covert_project_strategy.covert_fab_process_node'):
                continue

            # Set the value using dot notation
            try:
                _set_nested_attr(self, field_name, value)

            except AttributeError:
                # Field doesn't exist in Parameters - skip it (e.g., p_project_exists, p_fab_exists)
                pass

        # Handle process node mapping (special case because it has string values that map to enums)
        if 'covert_project_strategy.covert_fab_process_node' in data:
            process_node_map = {
                'best_indigenous': 'best_indigenous',
                'best_indigenous_gte_28nm': 'best_indigenous_gte_28nm',
                'best_indigenous_gte_14nm': 'best_indigenous_gte_14nm',
                'best_indigenous_gte_7nm': 'best_indigenous_gte_7nm',
                'best_available_indigenously': 'best_indigenous',
                'nm130': ProcessNode.nm130,
                'nm28': ProcessNode.nm28,
                'nm14': ProcessNode.nm14,
                'nm7': ProcessNode.nm7
            }
            self.covert_project_strategy.covert_fab_process_node = process_node_map.get(data['covert_project_strategy.covert_fab_process_node'], data['covert_project_strategy.covert_fab_process_node'])

    def to_dict(self):
        """
        Convert Parameters to a dictionary using dot notation for nested fields.
        Automatically flattens the nested dataclass structure.
        """
        def flatten_dataclass(obj, prefix=''):
            """Recursively flatten a dataclass to a dict with dot notation keys."""
            result = {}
            for field_name, field_value in obj.__dict__.items():
                full_key = f"{prefix}.{field_name}" if prefix else field_name

                # If it's a dataclass, recurse
                if hasattr(field_value, '__dataclass_fields__'):
                    result.update(flatten_dataclass(field_value, full_key))
                # Skip dict fields (like Probability_of_90p_PRC_localization_at_node)
                elif isinstance(field_value, dict):
                    continue
                # Convert enums to their value
                elif isinstance(field_value, Enum):
                    result[full_key] = field_value.value
                # Otherwise, add the value
                else:
                    result[full_key] = field_value

            return result

        return flatten_dataclass(self)

