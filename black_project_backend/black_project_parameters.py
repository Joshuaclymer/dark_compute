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
    agreement_start_year : Optional[int] = 2030
    num_years_to_simulate : float = 7.0  # Number of years from agreement start to simulate
    time_step_years : float = 0.1
    num_simulations : int = 200

    def validate(self):
        """Validate simulation settings parameters."""
        if self.agreement_start_year is not None:
            if self.agreement_start_year < 2026:
                raise ValueError(f"Agreement start year must be at least 2026 (got {self.agreement_start_year})")
            if self.agreement_start_year > 2031:
                raise ValueError(f"Agreement start year must be at most 2031 (got {self.agreement_start_year})")
        if self.num_years_to_simulate < 1:
            raise ValueError(f"Number of years to simulate must be at least 1 (got {self.num_years_to_simulate})")

@dataclass
class BlackProjectProperties:
    run_a_black_project : bool = True

    # Initial compute stock
    proportion_of_initial_compute_stock_to_divert : Optional[float] = 0.05

    # Data centers
    fraction_of_datacenter_capacity_not_built_for_concealment_diverted_to_black_project_at_agreement_start : float = 0.01
    datacenter_construction_labor : int = 10000
    years_before_agreement_year_prc_starts_building_black_datacenters : int = 1  # Years before agreement year to start building (0 = at agreement year)
    max_proportion_of_PRC_energy_consumption: float = 0.05

    # Covert fab
    build_a_black_fab : bool = True
    black_fab_operating_labor : Optional[int] = 550
    black_fab_construction_labor : Optional[int] = 250
    black_fab_process_node : Optional[ProcessNodeStrategy] = ProcessNodeStrategy.BEST_INDIGENOUS_GTE_28NM
    black_fab_proportion_of_prc_lithography_scanners_devoted : Optional[float] = 0.1

    # Researcher headcount
    researcher_headcount : int = 500

@dataclass
class DetectionParameters:
    mean_detection_time_for_100_workers: float = 6.95
    mean_detection_time_for_1000_workers: float = 3.42
    variance_of_detection_time_given_num_workers: float = 3.880
    us_intelligence_median_error_in_estimate_of_prc_compute_stock: float = 0.07
    us_intelligence_median_error_in_estimate_of_prc_datacenter_capacity: float = 0.01
    us_intelligence_median_error_in_estimate_of_prc_sme_stock: float = 0.07
    us_intelligence_median_error_in_energy_consumption_estimate_of_prc_datacenter_capacity: float = 0.05
    us_intelligence_median_error_in_satellite_estimate_of_prc_datacenter_capacity: float = 0.05

@dataclass
class ExogenousTrends:
    architecture_efficiency_improvement_per_year: float = 1.23 # moved from BlackFabParameters

    # moved from InitialPRCBlackComputeStockParameters
    total_prc_compute_stock_in_2025: float = 1e5
    annual_growth_rate_of_prc_compute_stock_p10: float = 1.3
    annual_growth_rate_of_prc_compute_stock_p50: float = 2.2
    annual_growth_rate_of_prc_compute_stock_p90: float = 3.0
    proportion_of_prc_chip_stock_produced_domestically_2026: float = 0.0
    proportion_of_prc_chip_stock_produced_domestically_2030: float = 0.7

    # Largest AI project compute trajectory (derived from AI Futures Project input_data.csv)
    # 2025 value: ~120,325 H100e, geometric mean growth rate 2025-2031: ~2.91x/year
    largest_ai_project_compute_stock_in_2025: float = 1.2e5
    annual_growth_rate_of_largest_ai_project_compute_stock: float = 2.91

    # Energy
    h100_power_watts: float = 700  # Moved from InitialPRCBlackComputeStockParameters 
    improvement_in_energy_efficiency_per_year: float = 1.26  # Moved from InitialPRCBlackComputeStockParameters
    energy_efficiency_of_prc_stock_relative_to_state_of_the_art: float = 0.2 # renamed from energy_efficiency_relative_to_h100: float = 0.5
    total_GW_of_PRC_energy_consumption: float = 1100

@dataclass
class SurvivalRateParameters:

    # Survival rate parameters
    initial_hazard_rate_p50: float = 0.01
    increase_of_hazard_rate_per_year_p50: float = 0.0035
    hazard_rate_p25_relative_to_p50: float = 0.1
    hazard_rate_p75_relative_to_p50: float = 6

@dataclass
class BlackDatacenterParameters:
    # -- Detection evidence --
    MW_per_construction_worker_per_year: float = 0.13
    relative_sigma_mw_per_construction_worker_per_year: float = 0.4

    operating_labor_per_MW: float = 1
    relative_sigma_operating_labor_per_MW: float = 0.4


@dataclass
class SlowdownCounterfactualParameters:
    """Parameters for counterfactual slowdown analysis"""
    fraction_of_prc_compute_spent_on_ai_rd_before_slowdown: float = 0.5

@dataclass
class BlackFabParameters:

    # Odds of covert project
    wafers_per_month_per_worker: float = 24.64
    labor_productivity_relative_sigma: float = 0.62
    wafers_per_month_per_lithography_scanner: float = 1000
    scanner_productivity_relative_sigma: float = 0.20

    # Localization probabilities (individual fields instead of dict for easier serialization)
    localization_130nm_2025: float = 0.80
    localization_130nm_2031: float = 0.80
    localization_28nm_2025: float = 0.0
    localization_28nm_2031: float = 0.6
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
class BlackProjectParameters:
    p_project_exists: float = 0.2
    survival_rate_parameters: SurvivalRateParameters = field(default_factory=SurvivalRateParameters)
    datacenter_model_parameters: BlackDatacenterParameters = field(default_factory=BlackDatacenterParameters)
    black_fab_parameters: BlackFabParameters = field(default_factory=BlackFabParameters)
    detection_parameters: DetectionParameters = field(default_factory=DetectionParameters)
    exogenous_trends: ExogenousTrends = field(default_factory=ExogenousTrends)
    slowdown_counterfactual_parameters: SlowdownCounterfactualParameters = field(default_factory=SlowdownCounterfactualParameters)

def _set_nested_attr(obj, path, value):
    """Set a nested attribute using dot notation: 'a.b.c' """
    parts = path.split('.')
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


@dataclass
class BlackProjectModelParameters:
    simulation_settings : SimulationSettings
    black_project_properties : BlackProjectProperties
    black_project_parameters : BlackProjectParameters

    def update_from_dict(self, data: dict):
        """
        Update parameters from request data using dot notation.
        Field names with dots (e.g., 'simulation_settings.agreement_start_year') are automatically
        converted to nested attribute access.
        Returns True if survival rate parameters changed (requires metalog cache clear).
        """
        for field_name, value in data.items():
            # Skip special cases handled below
            if field_name.startswith('black_project_properties.black_fab_process_node'):
                continue

            # Set the value using dot notation
            try:
                _set_nested_attr(self, field_name, value)

            except AttributeError:
                # Field doesn't exist in ModelParameters - skip it (e.g., p_project_exists, p_fab_exists)
                pass

        # Handle process node mapping (special case because it has string values that map to enums)
        if 'black_project_properties.black_fab_process_node' in data:
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
            self.black_project_properties.black_fab_process_node = process_node_map.get(data['black_project_properties.black_fab_process_node'], data['black_project_properties.black_fab_process_node'])

        # Validate parameters after update
        self.simulation_settings.validate()

    def to_dict(self):
        """
        Convert ModelParameters to a dictionary using dot notation for nested fields.
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
    
# Alias for backward compatibility
ModelParameters = BlackProjectModelParameters
