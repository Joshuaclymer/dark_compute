from dataclasses import dataclass
from typing import Optional, List
from enum import Enum
from fab_model import CovertFab, PRCCovertFab, ProcessNode
from stock_model import PRCDarkComputeStock
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from copy import deepcopy
from stock_model import Stock

# ============================================================================
# ENUMS
# ============================================================================

class UpdateSource(Enum):
    """Sources of intelligence that can update beliefs about covert projects"""
    PRC_COMPUTE_ACCOUNTING = "prc_compute_accounting"
    GLOBAL_COMPUTE_PRODUCTION_ACCOUNTING = "global_compute_production_accounting"
    FAB_INVENTORY_INTELLIGENCE = "fab_inventory_intelligence"
    FAB_PROCUREMENT_INTELLIGENCE = "fab_procurement_intelligence"
    FAB_OTHER_INTELLIGENCE = "fab_other_intelligence"  # HUMINT, SIGINT, etc.

# ============================================================================
# CLASSES
# ============================================================================

@dataclass
class CovertProjectStrategy:
    run_a_covert_project : bool

    # Initial compute stock
    proportion_of_initial_compute_stock_to_divert : Optional[float] = None

    # Covert fab
    build_a_covert_fab : bool = False
    covert_fab_operating_labor : Optional[int] = None
    covert_fab_construction_labor : Optional[int] = None
    covert_fab_process_node : Optional[ProcessNode] = None
    covert_fab_proportion_of_prc_lithography_scanners_devoted : Optional[float] = None

@dataclass
class CovertProject:
    name : str
    covert_project_strategy : CovertProjectStrategy
    agreement_year : float
    stock : Optional[Stock] = None
    covert_fab : Optional[CovertFab] = None

    def __post_init__(self):
        # Initialize covert fab if strategy requires it and construction_start_year is set
        self.dark_compute_stock = PRCDarkComputeStock(
            agreement_year = self.agreement_year,
            proportion_of_initial_compute_stock_to_divert = self.covert_project_strategy.proportion_of_initial_compute_stock_to_divert,
            optimal_proportion_of_initial_compute_stock_to_divert = best_prc_covert_project_strategy.proportion_of_initial_compute_stock_to_divert
        )

        if (self.covert_project_strategy.build_a_covert_fab):
            self.covert_fab = PRCCovertFab(
                construction_start_year = self.agreement_year,
                construction_labor = self.covert_project_strategy.covert_fab_construction_labor,
                process_node = self.covert_project_strategy.covert_fab_process_node,
                proportion_of_prc_lithography_scanners_devoted_to_fab = self.covert_project_strategy.covert_fab_proportion_of_prc_lithography_scanners_devoted,
                operation_labor = self.covert_project_strategy.covert_fab_operating_labor,
                agreement_year = self.agreement_year
            )


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
default_prc_covert_project_strategy = CovertProjectStrategy(
    run_a_covert_project = True,

    proportion_of_initial_compute_stock_to_divert = 0.05,

    # Covert fabs
    build_a_covert_fab = True,
    covert_fab_operating_labor = 550,
    covert_fab_construction_labor = 250,
    covert_fab_process_node = "best_available_indigenously",
    covert_fab_proportion_of_prc_lithography_scanners_devoted = 0.1,
)

best_prc_covert_project_strategy = default_prc_covert_project_strategy

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
                # ========== Section 1: Update beliefs about covert project (from dark compute stock accounting) ==========
                for detector_name, detector in self.detectors.items():
                    # Update from PRC compute accounting
                    prior_p_covert_project_exists = initial_priors[detector_name][project.name]['p_project_exists']
                    lr_prc = project.dark_compute_stock.lr_from_prc_compute_accounting
                    detector.beliefs_about_projects[project.name][current_year].update_p_project_exists(
                        year=current_year,
                        likelihood_ratio=lr_prc,
                        prior_p=prior_p_covert_project_exists,
                        source=UpdateSource.PRC_COMPUTE_ACCOUNTING
                    )

                    # Update from global compute production accounting
                    # Use the updated prior (which incorporates PRC compute accounting)
                    updated_prior_p = detector.beliefs_about_projects[project.name][current_year].p_project_exists
                    lr_global = project.dark_compute_stock.lr_from_global_compute_production_accounting
                    detector.beliefs_about_projects[project.name][current_year].update_p_project_exists(
                        year=current_year,
                        likelihood_ratio=lr_global,
                        prior_p=updated_prior_p,
                        source=UpdateSource.GLOBAL_COMPUTE_PRODUCTION_ACCOUNTING
                    )

                # ========== Section 2: Update beliefs about covert fab (from fab-specific intelligence) ==========
                if project.covert_fab is not None:
                    # h100e_produced_per_month returns monthly production, multiply by increment (in years) and months per year
                    additional_dark_compute = project.covert_fab.h100e_produced_per_month(current_year) * 12 * increment
                    project.dark_compute_stock.add_dark_compute(current_year, additional_dark_compute)

                    # Track cumulative covert fab production over time
                    # Get previous cumulative total or start at 0
                    prev_years = [y for y in project.covert_fab.dark_compute_over_time.keys() if y < current_year]
                    prev_total = project.covert_fab.dark_compute_over_time[max(prev_years)] if prev_years else 0
                    project.covert_fab.dark_compute_over_time[current_year] = prev_total + additional_dark_compute

                    # Update detector beliefs about covert fab
                    # detection_likelihood_ratio() automatically stores component LRs in the tracking dicts
                    likelihood_ratio = project.covert_fab.detection_likelihood_ratio(year=current_year)
                    project.covert_fab.detection_updates[current_year] = likelihood_ratio

                    # Get the three component LRs that were just calculated
                    lr_inventory = project.covert_fab.lr_inventory
                    lr_procurement = project.covert_fab.lr_procurement
                    lr_other = project.covert_fab.lr_other

                    for detector_name, detector in self.detectors.items():
                        # Update probability that covert fab exists (three separate updates)
                        prior_p_covert_fab_exists = initial_priors[detector_name][project.name]['p_covert_fab_exists']

                        # Update 1: Inventory intelligence
                        detector.beliefs_about_projects[project.name][current_year].update_p_covert_fab_exists(
                            year=current_year,
                            likelihood_ratio=lr_inventory,
                            prior_p=prior_p_covert_fab_exists,
                            source=UpdateSource.FAB_INVENTORY_INTELLIGENCE
                        )

                        # Update 2: Procurement intelligence (use updated prior from inventory)
                        updated_p_fab = detector.beliefs_about_projects[project.name][current_year].p_covert_fab_exists
                        detector.beliefs_about_projects[project.name][current_year].update_p_covert_fab_exists(
                            year=current_year,
                            likelihood_ratio=lr_procurement,
                            prior_p=updated_p_fab,
                            source=UpdateSource.FAB_PROCUREMENT_INTELLIGENCE
                        )

                        # Update 3: Other intelligence (use updated prior from procurement)
                        updated_p_fab = detector.beliefs_about_projects[project.name][current_year].p_covert_fab_exists
                        detector.beliefs_about_projects[project.name][current_year].update_p_covert_fab_exists(
                            year=current_year,
                            likelihood_ratio=lr_other,
                            prior_p=updated_p_fab,
                            source=UpdateSource.FAB_OTHER_INTELLIGENCE
                        )

                        # Also update probability that covert project exists (fab evidence implies project)
                        # Use the combined likelihood_ratio for project existence
                        prior_p_covert_project_exists = initial_priors[detector_name][project.name]['p_project_exists']

                        # Apply all three fab updates to project existence as well
                        detector.beliefs_about_projects[project.name][current_year].update_p_project_exists(
                            year=current_year,
                            likelihood_ratio=lr_inventory,
                            prior_p=prior_p_covert_project_exists,
                            source=UpdateSource.FAB_INVENTORY_INTELLIGENCE
                        )

                        updated_p_project = detector.beliefs_about_projects[project.name][current_year].p_project_exists
                        detector.beliefs_about_projects[project.name][current_year].update_p_project_exists(
                            year=current_year,
                            likelihood_ratio=lr_procurement,
                            prior_p=updated_p_project,
                            source=UpdateSource.FAB_PROCUREMENT_INTELLIGENCE
                        )

                        updated_p_project = detector.beliefs_about_projects[project.name][current_year].p_project_exists
                        detector.beliefs_about_projects[project.name][current_year].update_p_project_exists(
                            year=current_year,
                            likelihood_ratio=lr_other,
                            prior_p=updated_p_project,
                            source=UpdateSource.FAB_OTHER_INTELLIGENCE
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