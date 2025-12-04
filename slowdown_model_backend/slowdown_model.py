import numpy as np
import os
import sys

from classes.covert_prc_project import CovertPRCAIProject
from black_project_backend.classes.covert_fab import set_localization_probabilities
from classes.negotiator import Negotiator, BeliefsAboutProject, default_us_negotiation_strategy
from black_project_backend.black_project_parameters import CovertProjectProperties
from slowdown_model_paramaters import SlowdownModelParameters
from progress_model_incremental import ProgressModelIncremental
from progress_model import load_time_series_data, Parameters

# Add ai-futures-calculator scripts to path for sampling utilities
_AI_FUTURES_SCRIPTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'ai-futures-calculator', 'scripts')
if _AI_FUTURES_SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, _AI_FUTURES_SCRIPTS_PATH)
from python_sampling_bridge import load_config, sample_parameter_dict

# Path to the default sampling config
DEFAULT_SAMPLING_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'ai-futures-calculator', 'config', 'sampling_config.yaml'
)

# Path to the default time series data
_DEFAULT_TIME_SERIES_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'ai-futures-calculator', 'input_data.csv'
)

def load_default_time_series():
    """Load the default time series data from ai-futures-calculator."""
    return load_time_series_data(_DEFAULT_TIME_SERIES_PATH)

# ============================================================================
# SIMULATION: A simulation of a single rollout of a possible world
# ============================================================================

class SlowdownSimulation:

    def __init__(self, params: SlowdownModelParameters, set_parameters_to_medians: bool = False):
        self.params = params
        self.simulation_settings = params.slowdown_simulation_settings
        self.set_parameters_to_medians = set_parameters_to_medians

        # Store key years
        settings = params.slowdown_simulation_settings
        self.present_year = settings.present_year
        self.agreement_start_year = self._determine_agreement_start_year()
        self.simulation_end_year = settings.end_year

        # Initialize progress model
        progress_params = self._init_progress_params()
        initial_time_series = load_default_time_series()
        self.progress_model = ProgressModelIncremental(progress_params, initial_time_series)

        # Initialize covert projects and negotiators
        self.covert_projects = self._init_covert_projects()
        self.negotiators = self._init_negotiators()

    def _init_progress_params(self) -> Parameters:
        """Initialize progress model parameters.

        If set_parameters_to_medians is True, uses default parameters.
        Otherwise, Monte Carlo samples parameters from the sampling config.
        """
        if self.set_parameters_to_medians:
            # Use default parameters (which are the median values)
            return Parameters()

        # Monte Carlo sample parameters from the sampling config
        sampling_config = load_config(DEFAULT_SAMPLING_CONFIG_PATH)
        param_distributions = sampling_config.get('parameters', {})

        # Sample parameters using a random seed
        rng = np.random.default_rng()
        sampled_params = sample_parameter_dict(param_distributions, rng)

        # Create Parameters object with sampled values
        return Parameters(**sampled_params)

    def _determine_agreement_start_year(self) -> float:
        """Determine when the agreement starts based on simulation settings."""
        return self.simulation_settings.agreement_start_year or 2031

    def _init_negotiators(self) -> dict[str, Negotiator]:
        """Initialize negotiators with beliefs about covert projects."""
        return {
            "us_intelligence": Negotiator(
                name="us_intelligence",
                strategy=default_us_negotiation_strategy,
                beliefs_about_projects={
                    "prc_covert_project": BeliefsAboutProject(
                        p_project_exists=self.params.covert_project_parameters.p_project_exists,
                        initial_p_project_exists=self.params.covert_project_parameters.p_project_exists,
                        project_strategy_conditional_on_existence=CovertProjectProperties(),
                    )
                }
            )
        }

    def _init_covert_projects(self) -> dict[str, CovertPRCAIProject]:
        """Create covert projects with fresh random sampling."""
        # Update covert fab localization probabilities from parameters
        set_localization_probabilities(
            self.params.covert_project_parameters.covert_fab_parameters
        )

        # Calculate years array from present year to end year
        settings = self.params.slowdown_simulation_settings
        time_step = settings.time_step_years
        years = list(np.arange(settings.present_year, settings.end_year + time_step, time_step))

        return {
            "prc_covert_project": CovertPRCAIProject(
                name="prc_covert_project",
                covert_project_properties=self.params.covert_project_properties,
                agreement_year=self.agreement_start_year,
                years=years,
                covert_project_parameters=self.params.covert_project_parameters,
                progress_model=self.progress_model,
                set_parameters_to_medians=self.set_parameters_to_medians,
            )
        }

    def run_simulation(self):
        """Run the simulation from present year to simulation end year."""
        increment = self.simulation_settings.time_step_years

        # Run the simulation from the present year to the end year
        current_year = self.present_year + increment
        while current_year <= self.simulation_end_year:
            for project in self.covert_projects.values():

                # ========== Update compute stock and record compute/labor to histories ==========
                project.update_compute_and_labor(current_year, increment)

                # ========== Update AI progress model with operational dark compute ==========
                project.update_takeoff_progress(current_year)

                # ========== Get cumulative likelihood ratios from intelligence sources ==========
                project_lr = project.get_cumulative_evidence_of_covert_project(
                    current_year
                )

                # ========== Update negotiator beliefs with cumulative likelihood ratios ==========
                for _, negotiator in self.negotiators.items():
                    # Update project existence probability with cumulative LR
                    negotiator.beliefs_about_projects[project.name].update_p_project_exists_from_cumulative_lr(
                        year=current_year,
                        cumulative_likelihood_ratio=project_lr,
                    )

            current_year += increment
        return self.covert_projects, self.negotiators


# ============================================================================
# MODEL: Aggregates results across simulations
# ============================================================================

class SlowdownModel:
    def __init__(self, params: SlowdownModelParameters):
        """Initialize Model with a SlowdownModelParameters object.

        Args:
            params: SlowdownModelParameters object containing all simulation settings
        """
        self.parameters = params
        self.simulation_results = []

    def run_simulations(self, num_simulations: int):
        """Run multiple simulations.

        Args:
            num_simulations: Number of simulations to run
        """
        for _ in range(num_simulations):
            simulation = SlowdownSimulation(params=self.parameters)
            covert_projects, negotiators = simulation.run_simulation()
            self.simulation_results.append((covert_projects, negotiators))

    def run_median_simulation(self):
        """Run a single simulation with all parameters set to their median values.

        Returns:
            tuple: (covert_projects, negotiators) from the simulation
        """
        simulation = SlowdownSimulation(
            params=self.parameters,
            set_parameters_to_medians=True
        )
        covert_projects, negotiators = simulation.run_simulation()
        return covert_projects, negotiators