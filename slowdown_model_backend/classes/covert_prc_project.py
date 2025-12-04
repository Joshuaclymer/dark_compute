from typing import Optional
from black_project_backend.classes.covert_project import PRCBlackProject
from black_project_backend.classes.covert_fab import CovertFab
from black_project_backend.classes.black_project_stock import PRCBlackProjectStock, Compute
from black_project_backend.black_project_parameters import CovertProjectProperties, CovertProjectParameters
from slowdown_model_backend.classes.ai_project import AIProject
from progress_model_incremental import ProgressModelIncremental


class CovertPRCAIProject(PRCBlackProject, AIProject):
    """
    A covert PRC AI project that extends both PRCBlackProject and AIProject.

    This class combines:
    - PRCBlackProject's covert compute infrastructure (fabs, datacenters, dark compute stock)
    - AIProject's incremental progress model for tracking AI R&D speedup
    - Detection likelihood tracking

    Compute is measured in H100-equivalents (H100e), representing the number of
    H100 GPUs worth of compute capacity. The update() method automatically
    allocates operational dark compute across inference, experiments, and training
    according to configurable fractions.
    """

    # Default compute allocation percentages
    DEFAULT_EXPERIMENT_FRACTION = 0.80
    DEFAULT_INFERENCE_FRACTION = 0.10
    DEFAULT_TRAINING_FRACTION = 0.10

    def __init__(
        self,
        name: str,
        covert_project_properties: CovertProjectProperties,
        agreement_year: float,
        years: list,
        covert_project_parameters: CovertProjectParameters,
        progress_model: ProgressModelIncremental,
        stock: Optional[PRCBlackProjectStock] = None,
        covert_fab: Optional[CovertFab] = None,
        detection_time_via_other_strategies: Optional[float] = None,
        experiment_fraction: float = DEFAULT_EXPERIMENT_FRACTION,
        inference_fraction: float = DEFAULT_INFERENCE_FRACTION,
        training_fraction: float = DEFAULT_TRAINING_FRACTION,
        human_labor: float = 0.0,
        set_parameters_to_medians: bool = False,
    ):
        self.set_parameters_to_medians = set_parameters_to_medians

        # Initialize PRCBlackProject (handles covert infrastructure)
        PRCBlackProject.__init__(
            self,
            name=name,
            covert_project_properties=covert_project_properties,
            agreement_year=agreement_year,
            years=years,
            covert_project_parameters=covert_project_parameters,
            stock=stock,
            covert_fab=covert_fab,
            detection_time_via_other_strategies=detection_time_via_other_strategies,
            set_parameters_to_medians=set_parameters_to_medians,
        )

        # Initialize AIProject (handles progress model)
        AIProject.__init__(self, progress_model=progress_model)

        # Store compute allocation fractions (specific to this class)
        self.experiment_fraction = experiment_fraction
        self.inference_fraction = inference_fraction
        self.training_fraction = training_fraction
        self.human_labor = human_labor

    def update_compute_and_labor(self, current_year: float, increment: float) -> None:
        """
        Update compute stock from fab production and record compute/labor to histories.

        This method:
        1. Adds fab production to dark compute stock if fab exists
        2. Calculates operational compute and allocates it according to fractions
        3. Records human labor, inference, experiment, and training compute to histories

        Args:
            current_year: The current year in the simulation
            increment: Time step in years
        """
        # Add fab production to dark compute stock if fab exists
        if self.covert_fab is not None:
            compute_per_month = self.covert_fab.get_monthly_production_rate(current_year)
            months = 12 * increment
            scaled_chip_counts = {
                chip: count * months
                for chip, count in compute_per_month.chip_counts.items()
            }
            compute_to_add = Compute(chip_counts=scaled_chip_counts)
            self.black_project_stock.add_compute(current_year, compute_to_add)

        # Get operational compute (limited by datacenter capacity)
        operational_compute = self.get_operational_compute(current_year)

        # Get total H100-equivalent chips
        total_h100e = operational_compute.total_h100e_tpp()

        # Allocate compute according to fractions (all in H100e)
        inference_compute = total_h100e * self.inference_fraction
        experiment_compute = total_h100e * self.experiment_fraction
        training_compute = total_h100e * self.training_fraction

        # Record to histories (human_labor is constant for now)
        human_labor = 5.0  # TODO: Make this configurable
        self.human_labor_history.append(human_labor)
        self.inference_compute_history.append(inference_compute)
        self.experiment_compute_history.append(experiment_compute)
        self.training_compute_history.append(training_compute)
