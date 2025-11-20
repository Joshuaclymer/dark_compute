from dataclasses import dataclass, field
from typing import Optional, List
from copy import deepcopy

# ============================================================================
# CLASSES
# ============================================================================

# Import CovertProject and CovertProjectStrategy
from backend.classes.covert_project import CovertProject
from backend.classes.covert_fab import set_localization_probabilities
from backend.classes.dark_compute_stock import Compute
from backend.paramaters import CovertProjectStrategy, Parameters

@dataclass
class DetectorStrategy:
    placeholder : float # just leave this incomplete for now

@dataclass
class BeliefsAboutProject:
    p_project_exists : float = None
    p_covert_fab_exists : float = None
    project_strategy_conditional_on_existence : CovertProjectStrategy = None
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
class Detector:
    name : str
    strategy : DetectorStrategy
    beliefs_about_projects : dict[str, List[BeliefsAboutProject]]  # project name -> list of BeliefsAboutProject at each time step

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

        # Run the simulation from the agreement year to the end year
        current_year = self.year_us_prc_agreement_goes_into_force + increment
        while current_year <= end_year:
            for project in self.covert_projects.values():

                # ========== Add fab production to dark compute stock if fab exists ==========
                if project.covert_fab is not None:
                    # get_monthly_production_rate returns Compute object with monthly production rate, multiply by increment (in years) and months per year
                    compute_per_month = project.covert_fab.get_monthly_production_rate(current_year)
                    # Scale chip counts by the time increment to get total compute produced
                    months = 12 * increment
                    scaled_chip_counts = {chip: count * months for chip, count in compute_per_month.chip_counts.items()}
                    compute_to_add = Compute(chip_counts=scaled_chip_counts)
                    project.dark_compute_stock.add_dark_compute(current_year, compute_to_add)

                # ========== Get cumulative likelihood ratios from various intelligence sources ==========
                project_lr = project.get_cumulative_evidence_of_covert_project(current_year)

                # ========== Update detector beliefs with cumulative likelihood ratios ==========
                for detector_name, detector in self.detectors.items():
                    # Update project existence probability with cumulative LR
                    detector.beliefs_about_projects[project.name].update_p_project_exists_from_cumulative_lr(
                        year=current_year,
                        cumulative_likelihood_ratio=project_lr,
                    )
                
            current_year += increment
        return self.covert_projects, self.detectors


# ============================================================================
# MODEL: Aggregates results across simulations
# ============================================================================

class Model:
    def __init__(self, params: 'Parameters'):
        """Initialize Model with a Parameters object.

        Args:
            params: Parameters object containing all simulation settings
        """
        # Extract values from params
        self.parameters = params
        self.initial_detectors = {
            "us_intelligence" : Detector(
                name = "us_intelligence",
                strategy = default_us_detection_strategy,
                beliefs_about_projects = {
                    "prc_covert_project" : BeliefsAboutProject(
                        p_project_exists = self.parameters.covert_project_parameters.p_project_exists,
                        initial_p_project_exists = self.parameters.covert_project_parameters.p_project_exists,
                        project_strategy_conditional_on_existence = CovertProjectStrategy(),
                    )
                }
            )
        }

        self.simulation_results = []

    def _init_covert_projects(self):
        """Create a new set of covert projects with fresh random sampling.

        This method should be called for each simulation to ensure that
        random parameters (like detection times, localization years, etc.)
        are independently sampled for each simulation run.

        Returns:
            dict: Dictionary of covert project names to CovertProject instances
        """
        # Update covert fab localization probabilities from parameters
        set_localization_probabilities(self.parameters.covert_project_parameters.covert_fab_parameters)

        # Calculate years array from simulation settings
        import numpy as np
        start_year = self.parameters.simulation_settings.start_year
        end_year = self.parameters.simulation_settings.end_year
        time_step = self.parameters.simulation_settings.time_step_years
        years = list(np.arange(start_year, end_year + time_step, time_step))

        # Use the actual PRC strategy (not the US's beliefs about it)
        return {
            "prc_covert_project" : CovertProject(
                name = "prc_covert_project",
                covert_project_strategy = self.parameters.covert_project_strategy,
                agreement_year = self.parameters.simulation_settings.start_year,
                years = years,
                covert_project_parameters = self.parameters.covert_project_parameters
            )
        }

    def run_simulations(self, num_simulations : int):

        for _ in range(num_simulations):
            simulation = Simulation(
                year_us_prc_agreement_goes_into_force = self.parameters.simulation_settings.start_year,
                covert_projects = self._init_covert_projects(),  # Create fresh projects with new random sampling
                detectors = deepcopy(self.initial_detectors)
            )
            covert_projects, detectors = simulation.run_simulation(
                end_year = self.parameters.simulation_settings.end_year,
                increment = self.parameters.simulation_settings.time_step_years
            )
            self.simulation_results.append((covert_projects, detectors))