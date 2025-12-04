from dataclasses import dataclass
from typing import Optional

# ============================================================================
# CLASSES
# ============================================================================

# Import PRCBlackProject and BlackProjectProperties
from black_project_backend.classes.black_project import PRCBlackProject
from black_project_backend.classes.black_fab import set_localization_probabilities
from black_project_backend.classes.black_project_stock import Compute
from black_project_backend.black_project_parameters import BlackProjectProperties, ModelParameters, SimulationSettings

# ============================================================================
# SIMULATION: A simulation of a single rollout of a possible world
# ============================================================================

@dataclass
class BlackProjectSimulation:

    def __init__(
            self,
            black_projects: dict[str, PRCBlackProject],
            simulation_settings: SimulationSettings,
            agreement_start_year: float
        ):
        self.black_projects = black_projects
        self.simulation_settings = simulation_settings

        # Store agreement start year and calculate end year
        self.agreement_start_year = agreement_start_year
        self.simulation_end_year = agreement_start_year + simulation_settings.num_years_to_simulate

    def run_simulation(self):
        """Run the simulation from agreement start year to simulation end year."""
        increment = self.simulation_settings.time_step_years

        # Run the simulation from the agreement year to the end year
        current_year = self.agreement_start_year + increment
        while current_year <= self.simulation_end_year:
            for project in self.black_projects.values():

                # ========== Add fab production to dark compute stock if fab exists ==========
                if project.black_fab is not None:
                    # get_monthly_production_rate returns Compute object with monthly production rate, multiply by increment (in years) and months per year
                    compute_per_month = project.black_fab.get_monthly_production_rate(current_year)
                    # Scale chip counts by the time increment to get total compute produced
                    months = 12 * increment
                    scaled_chip_counts = {chip: count * months for chip, count in compute_per_month.chip_counts.items()}
                    compute_to_add = Compute(chip_counts=scaled_chip_counts)
                    project.black_project_stock.add_black_project(current_year, compute_to_add)

            current_year += increment
        return self.black_projects


# ============================================================================
# MODEL: Aggregates results across simulations
# ============================================================================

class BlackProjectModel:
    def __init__(self, params: 'ModelParameters'):
        """Initialize Model with a ModelParameters object.

        Args:
            params: ModelParameters object containing all simulation settings
        """
        # Extract values from params
        self.parameters = params
        self.simulation_results = []

    def _init_black_projects(self, agreement_year: float, simulation_end_year: float):
        """Create a new set of covert projects with fresh random sampling.

        This method should be called for each simulation to ensure that
        random parameters (like detection times, localization years, etc.)
        are independently sampled for each simulation run.

        Args:
            agreement_year: The year when the agreement starts
            simulation_end_year: The year when the simulation ends

        Returns:
            dict: Dictionary of covert project names to PRCBlackProject instances
        """
        # Update covert fab localization probabilities from parameters
        set_localization_probabilities(self.parameters.black_project_parameters.black_fab_parameters)

        # Calculate years array from simulation settings
        import numpy as np
        time_step = self.parameters.simulation_settings.time_step_years
        years = list(np.arange(agreement_year, simulation_end_year + time_step, time_step))

        # Use the actual PRC strategy (not the US's beliefs about it)
        return {
            "prc_black_project" : PRCBlackProject(
                name = "prc_black_project",
                black_project_properties = self.parameters.black_project_properties,
                agreement_year = agreement_year,
                years = years,
                black_project_parameters = self.parameters.black_project_parameters
            )
        }

    def run_simulations(self, num_simulations: int):
        """Run multiple simulations.

        Args:
            num_simulations: Number of simulations to run
        """
        # Determine agreement start year once (shared across simulations)
        agreement_year = self.parameters.simulation_settings.agreement_start_year
        simulation_end_year = agreement_year + self.parameters.simulation_settings.num_years_to_simulate

        for _ in range(num_simulations):
            simulation = BlackProjectSimulation(
                black_projects = self._init_black_projects(
                    agreement_year=agreement_year,
                    simulation_end_year=simulation_end_year
                ),
                simulation_settings = self.parameters.simulation_settings,
                agreement_start_year = agreement_year
            )
            black_projects = simulation.run_simulation()
            self.simulation_results.append(black_projects)
# Alias for backward compatibility
Model = BlackProjectModel
