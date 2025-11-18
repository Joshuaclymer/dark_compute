from dataclasses import dataclass
from typing import Optional, List
from enum import Enum
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from copy import deepcopy

# ============================================================================
# ENUMS
# ============================================================================

class UpdateSource(Enum):
    """Sources of intelligence that can update beliefs about covert projects"""
    PRC_COMPUTE_ACCOUNTING = "prc_compute_accounting"
    GLOBAL_COMPUTE_PRODUCTION_ACCOUNTING = "global_compute_production_accounting"
    DATACENTERS_CONCEALED = "datacenters_concealed"  # HUMINT, SIGINT, etc.
    FAB_INVENTORY_INTELLIGENCE = "fab_inventory_intelligence"
    FAB_PROCUREMENT_INTELLIGENCE = "fab_procurement_intelligence"
    FAB_OTHER_INTELLIGENCE = "fab_other_intelligence"  # HUMINT, SIGINT, etc.

# ============================================================================
# CLASSES
# ============================================================================

# Import CovertProject and CovertProjectStrategy from covert_project module
from covert_project import CovertProject, CovertProjectStrategy, best_prc_covert_project_strategy, default_prc_covert_project_strategy

@dataclass
class DetectorStrategy:
    placeholder : float # just leave this incomplete for now

@dataclass
class BeliefsAboutProject:
    p_project_exists : float
    p_covert_fab_exists : float
    project_strategy_conditional_on_existence : CovertProjectStrategy
    distribution_over_compute_operation : List[dict[List[float], List[float]]]  # List[year -> cumulative H100e produced]

    # Update history tracking: year -> list of dicts with 'update_size' and 'source'
    p_project_exists_update_history : dict = None
    p_covert_fab_exists_update_history : dict = None

    def __post_init__(self):
        if self.p_project_exists_update_history is None:
            self.p_project_exists_update_history = {}
        if self.p_covert_fab_exists_update_history is None:
            self.p_covert_fab_exists_update_history = {}

    def update_p_project_exists(self, year: float, likelihood_ratio: float, prior_p: float, source: UpdateSource):
        """Update p_project_exists using Bayesian update with likelihood ratio"""
        prior_odds = prior_p / (1 - prior_p)
        posterior_odds = prior_odds * likelihood_ratio
        posterior_p = posterior_odds / (1 + posterior_odds)

        # Store update history (append to list for this year)
        if year not in self.p_project_exists_update_history:
            self.p_project_exists_update_history[year] = []
        self.p_project_exists_update_history[year].append({
            'update_size': likelihood_ratio,
            'source': source
        })

        # Update the probability
        self.p_project_exists = posterior_p
        return posterior_p

    def update_p_covert_fab_exists(self, year: float, likelihood_ratio: float, prior_p: float, source: UpdateSource):
        """Update p_covert_fab_exists using Bayesian update with likelihood ratio"""
        prior_odds = prior_p / (1 - prior_p)
        posterior_odds = prior_odds * likelihood_ratio
        posterior_p = posterior_odds / (1 + posterior_odds)

        # Store update history (append to list for this year)
        if year not in self.p_covert_fab_exists_update_history:
            self.p_covert_fab_exists_update_history[year] = []
        self.p_covert_fab_exists_update_history[year].append({
            'update_size': likelihood_ratio,
            'source': source
        })

        # Update the probability
        self.p_covert_fab_exists = posterior_p
        return posterior_p

@dataclass
class Detector:
    name : str
    strategy : DetectorStrategy
    beliefs_about_projects : dict[str, List[BeliefsAboutProject]]  # project name -> BeliefsAboutProject at each time step

# ============================================================================
# DEFAULTS

default_us_detection_strategy = DetectorStrategy(
    placeholder = 0.0
)

# ============================================================================
# SIMULATION: A simulation of a single rollout of a possible world
# ============================================================================

@dataclass
class Simulation:

    def __init__(
            self,
            year_us_prc_agreement_goes_into_force : float,
            covert_projects: dict[str, CovertProject],
            detectors: dict[str, Detector]
        ):
        self.year_us_prc_agreement_goes_into_force = year_us_prc_agreement_goes_into_force
        self.covert_projects = covert_projects
        self.detectors = detectors

    def run_simulation(self, end_year : float, increment: float = 0.01):

        # Store initial priors to prevent them from being overwritten
        # The LRs are cumulative, so we always multiply by the initial prior
        initial_priors = {}
        for detector_name, detector in self.detectors.items():
            initial_priors[detector_name] = {}
            for project_name, beliefs_by_year in detector.beliefs_about_projects.items():
                if self.year_us_prc_agreement_goes_into_force in beliefs_by_year:
                    initial_priors[detector_name][project_name] = {
                        'p_covert_fab_exists': beliefs_by_year[self.year_us_prc_agreement_goes_into_force].p_covert_fab_exists,
                        'p_project_exists': beliefs_by_year[self.year_us_prc_agreement_goes_into_force].p_project_exists
                    }

        # Initialize beliefs for all years in the simulation
        current_year = self.year_us_prc_agreement_goes_into_force
        while current_year <= end_year:
            for detector in self.detectors.values():
                for project_name in self.covert_projects.keys():
                    if current_year not in detector.beliefs_about_projects[project_name]:
                        # Copy from the initial belief (only needed for current_year after the first)
                        initial_belief = detector.beliefs_about_projects[project_name][self.year_us_prc_agreement_goes_into_force]
                        detector.beliefs_about_projects[project_name][current_year] = BeliefsAboutProject(
                            p_project_exists=initial_belief.p_project_exists,
                            p_covert_fab_exists=initial_belief.p_covert_fab_exists,
                            project_strategy_conditional_on_existence=initial_belief.project_strategy_conditional_on_existence,
                            distribution_over_compute_operation=[]
                        )
            current_year += increment

        # Run the simulation from the agreement year to the end year
        current_year = self.year_us_prc_agreement_goes_into_force + increment
        while current_year <= end_year:
            for project in self.covert_projects.values():
                # ========== Add fab production to dark compute stock if fab exists ==========
                if project.covert_fab is not None:
                    # compute_produced_per_month returns Compute object with monthly production, multiply by increment (in years) and months per year
                    compute_per_month = project.covert_fab.compute_produced_per_month(current_year)
                    additional_dark_compute = compute_per_month.total_h100e_tpp() * 12 * increment
                    project.dark_compute_stock.add_dark_compute(current_year, additional_dark_compute)

                    # Track cumulative covert fab production over time
                    # Get previous cumulative total or start at 0
                    prev_years = [y for y in project.covert_fab.dark_compute_over_time.keys() if y < current_year]
                    prev_total = project.covert_fab.dark_compute_over_time[max(prev_years)] if prev_years else 0
                    project.covert_fab.dark_compute_over_time[current_year] = prev_total + additional_dark_compute

                    # Store detection updates for tracking (detection_likelihood_ratio is called inside get_aggregated_likelihood_ratios)
                    # We'll call it here to ensure the tracking dict is populated
                    project.covert_fab.detection_updates[current_year] = project.covert_fab.detection_likelihood_ratio(year=current_year)

                # ========== Get aggregated likelihood ratios from project ==========
                lr_results = project.get_aggregated_likelihood_ratios(current_year)
                project_lr = lr_results['project_lr']
                fab_lr = lr_results['fab_lr']
                sources = lr_results['sources']

                # ========== Update detector beliefs with aggregated likelihood ratios ==========
                for detector_name, detector in self.detectors.items():
                    # Get initial priors for this detector
                    prior_p_project = initial_priors[detector_name][project.name]['p_project_exists']
                    prior_p_fab = initial_priors[detector_name][project.name]['p_covert_fab_exists']

                    # Update project existence probability with aggregated LR
                    detector.beliefs_about_projects[project.name][current_year].update_p_project_exists(
                        year=current_year,
                        likelihood_ratio=project_lr,
                        prior_p=prior_p_project,
                        source=UpdateSource.PRC_COMPUTE_ACCOUNTING  # Use first source as representative
                    )

                    # Update fab existence probability if fab exists
                    if project.covert_fab is not None:
                        detector.beliefs_about_projects[project.name][current_year].update_p_covert_fab_exists(
                            year=current_year,
                            likelihood_ratio=fab_lr,
                            prior_p=prior_p_fab,
                            source=UpdateSource.FAB_INVENTORY_INTELLIGENCE  # Use first fab source as representative
                        )

            current_year += increment
        return self.covert_projects, self.detectors


# ============================================================================
# MODEL: Aggregates results across simulations
# ============================================================================

class Model:
    def __init__(
            self,
            year_us_prc_agreement_goes_into_force : float,
            end_year : float,
            increment: float,
            prc_strategy: 'CovertProjectStrategy' = None,
        ):
        self.year_us_prc_agreement_goes_into_force = year_us_prc_agreement_goes_into_force
        self.end_year = end_year
        self.increment = increment

        # Store the PRC's actual strategy (what they actually do)
        # Default to best_prc_covert_project_strategy if not provided
        self.prc_strategy = prc_strategy if prc_strategy is not None else best_prc_covert_project_strategy

        self.initial_detectors = {
            "us_intelligence" : Detector(
                name = "us_intelligence",
                strategy = default_us_detection_strategy,
                beliefs_about_projects = {
                    "prc_covert_project" : {
                        year_us_prc_agreement_goes_into_force:
                            BeliefsAboutProject(
                            p_project_exists = 0.2,
                            p_covert_fab_exists = 0.1,
                            # US beliefs about PRC strategy are always best_prc_covert_project_strategy
                            project_strategy_conditional_on_existence = best_prc_covert_project_strategy,
                            distribution_over_compute_operation = []
                    )
                }}
            )
        }

        self.simulation_results = []

    def _create_fresh_covert_projects(self):
        """Create a new set of covert projects with fresh random sampling.

        This method should be called for each simulation to ensure that
        random parameters (like detection times, localization years, etc.)
        are independently sampled for each simulation run.

        Returns:
            dict: Dictionary of covert project names to CovertProject instances
        """
        # Use the actual PRC strategy (not the US's beliefs about it)
        return {
            "prc_covert_project" : CovertProject(
                name = "prc_covert_project",
                covert_project_strategy = self.prc_strategy,
                agreement_year = self.year_us_prc_agreement_goes_into_force,
            )
        }

    def run_simulations(self, num_simulations : int):

        for _ in range(num_simulations):
            simulation = Simulation(
                year_us_prc_agreement_goes_into_force = self.year_us_prc_agreement_goes_into_force,
                covert_projects = self._create_fresh_covert_projects(),  # Create fresh projects with new random sampling
                detectors = deepcopy(self.initial_detectors)
            )
            covert_projects, detectors = simulation.run_simulation(
                end_year = self.end_year,
                increment = self.increment
            )
            self.simulation_results.append((covert_projects, detectors))