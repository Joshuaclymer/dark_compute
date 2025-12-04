from dataclasses import dataclass, field
from typing import List

from black_project_backend.black_project_parameters import CovertProjectProperties


@dataclass
class NegotiatorStrategy:
    placeholder : float # just leave this incomplete for now

@dataclass
class BeliefsAboutProject:
    p_project_exists : float = None
    p_covert_fab_exists : float = None
    project_strategy_conditional_on_existence : 'CovertProjectProperties' = None
    distribution_over_compute_operation : list = field(default_factory=list)
    p_project_exists_update_history : dict = field(default_factory=dict)
    initial_p_project_exists : float = None  # Store the initial prior

    def update_p_project_exists_from_cumulative_lr(self, year: float, cumulative_likelihood_ratio: float):
        """Update p_project_exists using cumulative likelihood ratio from initial prior"""
        # Always start from the initial prior
        if self.initial_p_project_exists is None:
            self.initial_p_project_exists = self.p_project_exists

        # Clamp initial prior to avoid division by zero
        p_clamped = max(1e-10, min(1 - 1e-10, self.initial_p_project_exists))
        prior_odds = p_clamped / (1 - p_clamped)
        posterior_odds = prior_odds * cumulative_likelihood_ratio
        posterior_p = posterior_odds / (1 + posterior_odds)

        # Store update history (append to list for this year)
        if year not in self.p_project_exists_update_history:
            self.p_project_exists_update_history[year] = []
        self.p_project_exists_update_history[year].append({
            'update': cumulative_likelihood_ratio,
            'current_p_project_exists': posterior_p
        })

        # Update the current p_project_exists
        self.p_project_exists = posterior_p

        return posterior_p


@dataclass
class Negotiator:
    name : str
    strategy : NegotiatorStrategy
    beliefs_about_projects : dict[str, List[BeliefsAboutProject]]  # project name -> list of BeliefsAboutProject at each time step

# ============================================================================
# DEFAULTS

default_us_negotiation_strategy = NegotiatorStrategy(
    placeholder = 0.0
)