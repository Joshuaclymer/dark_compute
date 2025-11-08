from dataclasses import dataclass
from typing import Optional, List
from fab_model import CovertFab, PRCCovertFab, ProcessNode
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from copy import deepcopy

# ============================================================================
# CLASSES
# ============================================================================

@dataclass
class CovertProjectStrategy:
    run_a_covert_project : bool

    # Covert fab
    build_a_covert_fab : bool
    covert_fab_operating_labor : Optional[int]
    covert_fab_construction_labor : Optional[int]
    covert_fab_process_node : Optional[ProcessNode]
    covert_fab_proportion_of_prc_lithography_scanners_devoted : Optional[float]

@dataclass
class CovertProject:
    name : str
    h100e_over_time : dict[List[float], List[float]]  # year -> cumulative H100e produced
    covert_project_strategy : CovertProjectStrategy
    agreement_year : float
    construction_start_year : Optional[float] = None
    covert_fab : Optional[CovertFab] = None

    def __post_init__(self):
        # Initialize covert fab if strategy requires it and construction_start_year is set
        if (self.covert_project_strategy.build_a_covert_fab and
            self.construction_start_year is not None):
            self.covert_fab = PRCCovertFab(
                construction_start_year = self.construction_start_year,
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
    build_a_covert_fab = True,
    covert_fab_operating_labor = 728,
    covert_fab_construction_labor = 448,
    covert_fab_process_node = "best_available_indigenously",
    covert_fab_proportion_of_prc_lithography_scanners_devoted = 0.102,
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

        # Run the simulation from the agreement year to the end year
        current_year = self.year_us_prc_agreement_goes_into_force
        while current_year <= end_year:
            for project in self.covert_projects.values():
                current_h100e = project.h100e_over_time.get(current_year - increment, 0.0)

                # Make updates pertaining to covert fabs
                if project.covert_fab is not None:
                    # h100e_produced_per_month returns monthly production, multiply by increment (in years) and months per year
                    updated_h100e = project.covert_fab.h100e_produced_per_month(current_year) * 12 * increment
                    project.h100e_over_time[current_year] = current_h100e + updated_h100e
                    # Update detector beliefs
                    likelihood_ratio = project.covert_fab.detection_likelihood_ratio(year=current_year)
                    for detector_name, detector in self.detectors.items():
                        # Use stored initial priors (LRs are cumulative, so always multiply by initial prior)
                        prior_p_covert_fab_exists = initial_priors[detector_name][project.name]['p_covert_fab_exists']
                        prior_p_covert_project_exists = initial_priors[detector_name][project.name]['p_project_exists']
                        updated_odds_of_covert_fab = prior_p_covert_fab_exists / (1 - prior_p_covert_fab_exists) * likelihood_ratio
                        updated_p_covert_fab_exists = updated_odds_of_covert_fab / (1 + updated_odds_of_covert_fab)
                        updated_odds_of_covert_project = prior_p_covert_project_exists / (1 - prior_p_covert_project_exists) * likelihood_ratio
                        updated_p_covert_project_exists = updated_odds_of_covert_project / (1 + updated_odds_of_covert_project)
                        detector.beliefs_about_projects[project.name][current_year] = BeliefsAboutProject(
                            p_project_exists=updated_p_covert_project_exists,
                            p_covert_fab_exists=updated_p_covert_fab_exists,
                            project_strategy_conditional_on_existence=detector.beliefs_about_projects[project.name][self.year_us_prc_agreement_goes_into_force].project_strategy_conditional_on_existence,
                            distribution_over_compute_operation=[]
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
                h100e_over_time = {self.year_us_prc_agreement_goes_into_force: 0.0},
                covert_project_strategy = self.prc_strategy,
                agreement_year = self.year_us_prc_agreement_goes_into_force,
                construction_start_year = self.year_us_prc_agreement_goes_into_force
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