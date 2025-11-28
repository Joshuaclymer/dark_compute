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
    start_agreement_at_what_ai_rnd_speedup: Optional[float] = None # can also be a milestone like "AC" or "SAR" or None if starting agreement at a specific year
    start_agreement_at_specific_year : Optional[int] = 2030
    num_years_to_simulate : float = 7.0  # Number of years from agreement start to simulate
    time_step_years : float = 0.1
    num_simulations : int = 60

    def validate(self):
        """Validate simulation settings parameters."""
        if self.start_agreement_at_specific_year is not None:
            if self.start_agreement_at_specific_year < 2026:
                raise ValueError(f"Agreement start year must be at least 2026 (got {self.start_agreement_at_specific_year})")
            if self.start_agreement_at_specific_year > 2031:
                raise ValueError(f"Agreement start year must be at most 2031 (got {self.start_agreement_at_specific_year})")
        if self.num_years_to_simulate < 1:
            raise ValueError(f"Number of years to simulate must be at least 1 (got {self.num_years_to_simulate})")

@dataclass
class CovertProjectProperties:
    run_a_covert_project : bool = True

    # Initial compute stock
    proportion_of_initial_compute_stock_to_divert : Optional[float] = 0.05

    # Data centers
    datacenter_construction_labor : int = 10000
    years_before_agreement_year_prc_starts_building_covert_datacenters : int = 1  # Years before agreement year to start building (0 = at agreement year)

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
    total_prc_compute_stock_in_2025: float = 1e5
    energy_efficiency_relative_to_h100: float = 0.5
    annual_growth_rate_of_prc_compute_stock_p10: float = 1.3
    annual_growth_rate_of_prc_compute_stock_p50: float = 2.2
    annual_growth_rate_of_prc_compute_stock_p90: float = 3.0
    proportion_of_prc_chip_stock_produced_domestically_2026: float = 0.0
    proportion_of_prc_chip_stock_produced_domestically_2030: float = 0.7

    us_intelligence_median_error_in_estimate_of_prc_compute_stock: float = 0.07

@dataclass
class SlowdownCounterfactualParameters:
    # Slowdown counterfactual parameters (for reference line in plots)
    # Note: Global AI R&D compute is now sourced from AI Futures Project input_data.csv
    fraction_of_prc_compute_spent_on_ai_rd_before_slowdown: float = 0.5

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
    MW_per_construction_worker_per_year: float = 0.2
    relative_sigma_mw_per_construction_worker_per_year: float = 0.4

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
    slowdown_counterfactual_parameters: SlowdownCounterfactualParameters = field(default_factory=SlowdownCounterfactualParameters)
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
class ModelParameters:
    simulation_settings : SimulationSettings
    covert_project_properties : CovertProjectProperties
    covert_project_parameters : CovertProjectParameters

    def update_from_dict(self, data: dict):
        """
        Update parameters from request data using dot notation.
        Field names with dots (e.g., 'simulation_settings.start_agreement_at_specific_year') are automatically
        converted to nested attribute access.
        Returns True if survival rate parameters changed (requires metalog cache clear).
        """
        for field_name, value in data.items():
            # Skip special cases handled below
            if field_name.startswith('covert_project_properties.covert_fab_process_node'):
                continue

            # Set the value using dot notation
            try:
                _set_nested_attr(self, field_name, value)

            except AttributeError:
                # Field doesn't exist in ModelParameters - skip it (e.g., p_project_exists, p_fab_exists)
                pass

        # Handle process node mapping (special case because it has string values that map to enums)
        if 'covert_project_properties.covert_fab_process_node' in data:
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
            self.covert_project_properties.covert_fab_process_node = process_node_map.get(data['covert_project_properties.covert_fab_process_node'], data['covert_project_properties.covert_fab_process_node'])

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
    
@dataclass
class PCatastropheParameters:
    p_ai_takeover_t1: float = 0.40 # time is adjusted for safety research speed
    p_ai_takeover_t2: float = 0.15
    p_ai_takeover_t3: float = 0.05

    p_human_power_grabs_t1: float = 0.40 # time is NOT adjusted for research speed
    p_human_power_grabs_t2: float = 0.20
    p_human_power_grabs_t3: float = 0.10

    # Safety research speedup = capability_speedup ^ safety_speedup_exponent
    # e.g., exponent=0.5 means safety speedup is sqrt of capability speedup
    safety_speedup_exponent: float = 0.5

@dataclass
class ProxyProject:
    compute_cap_as_percentile_of_PRC_operational_covert_compute: float = 0.7 #70th percentile
    frequency_cap_is_updated_in_years: float = 1.0

@dataclass
class SoftwareProliferation:
    weight_stealing_times: list = field(default_factory=lambda: ["SC"]) #Options: "SC", "SAR", 2027, or 2030
    stealing_algorithms_up_to: str = "SAR" # Options: "SC", "SAR"

@dataclass
class SlowdownPageParameters:
    monte_carlo_samples: int = 1
    ai_rnd_speedup_at_agreement_start: float = 2.0
    PCatastrophe_parameters: PCatastropheParameters = field(default_factory=PCatastropheParameters)
    proxy_project: ProxyProject = field(default_factory=ProxyProject)
    software_proliferation: SoftwareProliferation = field(default_factory=SoftwareProliferation)