from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from black_project_backend.black_project_parameters import (
    CovertProjectProperties,
    CovertProjectParameters,
    ProcessNode,
    _set_nested_attr,
)

@dataclass
class SlowdownSimulationSettings:
    start_agreement_at_what_ai_rnd_speedup: Optional[float] = None # can also be a milestone like "AC" or "SAR" or None if starting agreement at a specific year
    start_agreement_at_specific_year : Optional[int] = 2030
    num_years_to_simulate : float = 7.0  # Number of years from agreement start to simulate
    time_step_years : float = 0.1
    num_simulations : int = 200
    present_year : int = 2026
    end_year : int = 2040

@dataclass
class TakeoverRiskParameters:
    # --- P(misalignment at handoff) anchor points ---
    # These are probabilities at 1 month, 1 year, and 10 years of adjusted alignment research time
    # See slowdown_model.md for details on how adjusted alignment research time is computed
    p_misalignment_at_handoff_t1: float = 0.40  # at 1 month of adjusted research time
    p_misalignment_at_handoff_t2: float = 0.10  # at 1 year
    p_misalignment_at_handoff_t3: float = 0.04  # at 10 years

    # --- P(human power grabs) anchor points ---
    # Based on SAR to ASI duration (time society has to react)
    # See slowdown_model.md section "Modeling Human Power Grab Risk"
    p_human_power_grabs_t1: float = 0.40  # at 1 month SAR-to-ASI
    p_human_power_grabs_t2: float = 0.15  # at 1 year
    p_human_power_grabs_t3: float = 0.05  # at 10 years

    # --- Safety research adjustment parameters ---
    # Safety research speedup = capability_speedup * safety_speedup_multiplier
    # e.g., multiplier=0.5 means alignment speedup is half of capability speedup
    safety_speedup_multiplier: float = 0.5

    # Maximum alignment speedup before handoff
    # Caps the alignment speedup during pre-AC and AC-to-handoff periods
    # Alignment speedup = min(max_alignment_speedup_before_handoff, capability_speedup * safety_speedup_multiplier)
    max_alignment_speedup_before_handoff: float = 5.0

    # Backward compatibility aliases for p_ai_takeover (deprecated naming)
    @property
    def p_ai_takeover_t1(self) -> float:
        return self.p_misalignment_at_handoff_t1

    @property
    def p_ai_takeover_t2(self) -> float:
        return self.p_misalignment_at_handoff_t2

    @property
    def p_ai_takeover_t3(self) -> float:
        return self.p_misalignment_at_handoff_t3

    # Discount factor for pre-handoff research (present day -> automated coder)
    # Research during this period is less relevant than during handoff window
    research_relevance_of_pre_handoff_discount: float = 0.1

    # Multiplier for alignment research effort during slowdown period
    # If > 1, indicates increased focus on alignment during slowdown
    increase_in_alignment_research_effort_during_slowdown: float = 1.5

    # --- Post-handoff parameters ---
    # Present day year for calculations
    present_day_year: float = 2026.0

    # --- P(misalignment after handoff) anchor points ---
    # These are probabilities at different alignment tax values (proportion of compute spent on alignment)
    # The alignment tax is the fraction of compute devoted to alignment after handoff (0 to 1)
    # t1 = 1% alignment tax, t2 = 10% alignment tax, t3 = 100% alignment tax
    p_misalignment_after_handoff_t1: float = 0.3  # at 1% alignment tax paid
    p_misalignment_after_handoff_t2: float = 0.15  # at 10% alignment tax paid
    p_misalignment_after_handoff_t3: float = 0.1  # at 100% alignment tax paid

    # Alignment tax after handoff: the proportion of compute spent on alignment (0 to 1)
    # This is a user-specified input, not computed from the trajectory
    alignment_tax_after_handoff: float = 0.10  # default 10% of compute on alignment


@dataclass
class SoftwareProliferationParameters:
    weight_stealing_times: list = field(default_factory=lambda: ["SC"]) #Options: "SC", "SAR", 2027, or 2030
    stealing_algorithms_up_to: str = "SAR" # Options: "SC", "SAR"

@dataclass
class ProxyProjectParameters:
    """Parameters for the US slowdown trajectory (compute cap based on PRC covert compute)."""
    compute_cap_as_percentile_of_PRC_operational_covert_compute: float = 0.7 #70th percentile
    frequency_cap_is_updated_in_years: float = 1.0
    determine_optimal_proxy_project_compute_based_on_risk_curves : bool = False


@dataclass
class USProjectParameters:
    """Parameters for capability cap trajectory"""
    years_after_agreement_start_when_evaluation_based_capability_cap_is_implemented: float = 0.5
    capability_cap_ai_randd_speedup: float = TakeoverRiskParameters.max_alignment_speedup_before_handoff / TakeoverRiskParameters.safety_speedup_multiplier # Options: "AC", "SAR", or year like "2028"

@dataclass
class SlowdownModelParameters:
    # Fields without defaults must come first
    covert_project_properties : CovertProjectProperties = field(default_factory=CovertProjectProperties)
    covert_project_parameters : CovertProjectParameters = field(default_factory=CovertProjectParameters)
    # Fields with defaults
    slowdown_simulation_settings: SlowdownSimulationSettings = field(default_factory=SlowdownSimulationSettings)
    takeover_risk: TakeoverRiskParameters = field(default_factory=TakeoverRiskParameters)
    proxy_project: ProxyProjectParameters = field(default_factory=ProxyProjectParameters)
    software_proliferation: SoftwareProliferationParameters = field(default_factory=SoftwareProliferationParameters)
    us_project: USProjectParameters = field(default_factory=USProjectParameters)

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
    
