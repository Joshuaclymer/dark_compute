from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
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

# Import CovertProject and CovertProjectStrategy
from backend.classes.covert_project import CovertProject
from backend.paramaters import CovertProjectStrategy

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

    def update_p_project_exists(self, year: float, likelihood_ratio: float):
        """Update p_project_exists using Bayesian update with likelihood ratio"""
        # Clamp p_project_exists to avoid division by zero
        p_clamped = max(1e-10, min(1 - 1e-10, self.p_project_exists))
        prior_odds = p_clamped / (1 - p_clamped)
        posterior_odds = prior_odds * likelihood_ratio
        posterior_p = posterior_odds / (1 + posterior_odds)

        # Store update history (append to list for this year)
        if year not in self.p_project_exists_update_history:
            self.p_project_exists_update_history[year] = []
        self.p_project_exists_update_history[year].append({
            'update': likelihood_ratio,
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
                    # compute_produced_per_month returns Compute object with monthly production, multiply by increment (in years) and months per year
                    compute_per_month = project.covert_fab.compute_produced_per_month(current_year)
                    additional_dark_compute = compute_per_month.total_h100e_tpp() * 12 * increment
                    project.dark_compute_stock.add_dark_compute(current_year, additional_dark_compute)

                # ========== Get cumulative likelihood ratios from various intelligence sources ==========
                project_lr = project.get_cumulative_evidence_of_covert_project(current_year)

                # ========== Update detector beliefs with cumulative likelihood ratios ==========
                for detector_name, detector in self.detectors.items():
                    # Update project existence probability with aggregated LR
                    detector.beliefs_about_projects[project.name].update_p_project_exists(
                        year=current_year,
                        likelihood_ratio=project_lr,
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
            p_project_exists: float = 0.2
        ):
        self.year_us_prc_agreement_goes_into_force = year_us_prc_agreement_goes_into_force
        self.end_year = end_year
        self.increment = increment

        # Store the PRC's actual strategy (what they actually do)
        # Default to CovertProjectStrategy() with defaults if not provided
        self.prc_strategy = prc_strategy if prc_strategy is not None else CovertProjectStrategy()

        self.initial_detectors = {
            "us_intelligence" : Detector(
                name = "us_intelligence",
                strategy = default_us_detection_strategy,
                beliefs_about_projects = {
                    "prc_covert_project" : BeliefsAboutProject(
                        p_project_exists = p_project_exists,
                        p_covert_fab_exists = 0.1,
                        # US beliefs about PRC strategy use default CovertProjectStrategy
                        project_strategy_conditional_on_existence = CovertProjectStrategy(),
                        distribution_over_compute_operation = []
                    )
                }
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