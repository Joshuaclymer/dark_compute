from typing import Optional
from black_project_backend.classes.covert_fab import CovertFab, PRCCovertFab, ProcessNode, FabNotBuiltException, Compute
from black_project_backend.classes.black_project_stock import PRCBlackProjectStock
from black_project_backend.classes.covert_datacenters import CovertPRCDatacenters
from black_project_backend.util import lr_over_time_vs_num_workers
from black_project_backend.black_project_parameters import CovertProjectProperties, CovertProjectParameters


class CovertPRCInfrastructure:
    """
    A covert PRC infrastructure with compute infrastructure including fabs, datacenters, and dark compute stock.
    """

    def __init__(
        self,
        name: str,
        covert_project_properties: CovertProjectProperties,
        agreement_year: float,
        years: list,
        covert_project_parameters: CovertProjectParameters,
        stock: Optional[PRCBlackProjectStock] = None,
        covert_fab: Optional[CovertFab] = None,
        detection_time_via_other_strategies: Optional[float] = None,
        set_parameters_to_medians: bool = False,
    ):
        self.name = name
        self.covert_project_properties = covert_project_properties
        self.agreement_year = agreement_year
        self.years = years
        self.covert_project_parameters = covert_project_parameters
        self.stock = stock
        self.covert_fab = covert_fab
        self.detection_time_via_other_strategies = detection_time_via_other_strategies
        self.set_parameters_to_medians = set_parameters_to_medians

        # Initialize covert infrastructure
        self._init_covert_infrastructure()

    def _init_covert_infrastructure(self):
        """Initialize all covert infrastructure components."""
        self.black_project_stock = PRCBlackProjectStock(
            agreement_year=self.agreement_year,
            proportion_of_initial_compute_stock_to_divert=self.covert_project_properties.proportion_of_initial_compute_stock_to_divert,
            optimal_proportion_of_initial_compute_stock_to_divert=self.covert_project_properties.proportion_of_initial_compute_stock_to_divert,
            initial_compute_parameters=self.covert_project_parameters.initial_compute_stock_parameters,
            survival_parameters=self.covert_project_parameters.survival_rate_parameters
        )

        # Convert absolute years to years since agreement start
        years_since_agreement_start = [year - self.agreement_year for year in self.years]

        # Calculate datacenter start year offset (relative to agreement year)
        # years_before is positive if construction starts before agreement, negative if after
        years_before = self.covert_project_properties.years_before_agreement_year_prc_starts_building_covert_datacenters
        datacenter_start_year_offset = -years_before

        self.covert_datacenters = CovertPRCDatacenters(
            construction_labor=self.covert_project_properties.datacenter_construction_labor,
            years_since_agreement_start=years_since_agreement_start,
            datacenter_parameters=self.covert_project_parameters.datacenter_model_parameters,
            project_parameters=self.covert_project_parameters,
            datacenter_start_year_offset=datacenter_start_year_offset
        )

        if self.covert_project_properties.build_a_covert_fab:
            try:
                self.covert_fab = PRCCovertFab(
                    construction_start_year=self.agreement_year,
                    construction_labor=self.covert_project_properties.covert_fab_construction_labor,
                    process_node=self.covert_project_properties.covert_fab_process_node,
                    proportion_of_prc_lithography_scanners_devoted_to_fab=self.covert_project_properties.covert_fab_proportion_of_prc_lithography_scanners_devoted,
                    operation_labor=self.covert_project_properties.covert_fab_operating_labor,
                    agreement_year=self.agreement_year,
                    years_since_agreement_start=years_since_agreement_start,
                    project_parameters=self.covert_project_parameters
                )
            except FabNotBuiltException:
                # Fab not built because process node threshold not met
                self.covert_fab = None

        # Precompute lr_over_time_vs_num_workers for the total project
        labor_by_year = {}
        for year in years_since_agreement_start:
            # Calculate total labor at this specific year
            labor_at_year = self.covert_datacenters.construction_labor
            labor_at_year += self.covert_datacenters.get_operating_labor(year)
            # Add fab labor if it exists
            if self.covert_fab is not None:
                labor_at_year += self.covert_project_properties.covert_fab_construction_labor
                labor_at_year += self.covert_project_properties.covert_fab_operating_labor
            labor_by_year[year] = int(labor_at_year)

        self.lr_over_time_vs_num_workers = lr_over_time_vs_num_workers(
            labor_by_year=labor_by_year,
            mean_detection_time_100_workers=self.covert_project_parameters.mean_detection_time_for_100_workers,
            mean_detection_time_1000_workers=self.covert_project_parameters.mean_detection_time_for_1000_workers,
            variance_theta=self.covert_project_parameters.variance_of_detection_time_given_num_workers
        )

    def operational_black_project(self, year: float):
        """Calculate operational dark compute limited by datacenter energy capacity.

        This gets all surviving dark compute and scales it down if the energy requirements
        exceed the available datacenter capacity.

        Args:
            year: The year to calculate operational compute for

        Returns:
            Compute object containing chips that can be powered with available capacity
        """
        # Get datacenter capacity from covert_datacenters
        year_since_agreement = year - self.agreement_year
        datacenter_capacity_gw = self.covert_datacenters.get_GW_capacity(year_since_agreement)

        # Get all surviving dark compute (using absolute year, not year_since_agreement)
        all_black_project = self.black_project_stock.black_project(year)

        # Calculate total energy requirements
        total_energy_required_gw = all_black_project.total_energy_requirements_GW()

        # If energy requirements are within capacity, return all dark compute
        if total_energy_required_gw <= datacenter_capacity_gw:
            return all_black_project

        # Otherwise, scale down chip counts proportionally to fit within capacity
        scaling_factor = datacenter_capacity_gw / total_energy_required_gw

        # Create new chip_counts with scaled values
        operational_chip_counts = {
            chip: count * scaling_factor
            for chip, count in all_black_project.chip_counts.items()
        }

        operational_compute = Compute(chip_counts=operational_chip_counts)

        return operational_compute

    def h100_years_to_date(self, current_year: float, years_list: list[float]) -> float:
        """Calculate total H100-years of computation performed up to the current year.

        This integrates operational H100e over time from agreement start until the current year,
        using the operational_black_project method to get capacity-limited compute at each timestep.

        Args:
            current_year: The year to calculate H100-years up to
            years_list: List of years to integrate over (should include agreement_year through current_year)

        Returns:
            float: Total H100-years of computation performed (H100e x years)
        """
        # Filter years from agreement start up to current year
        years = sorted([y for y in years_list if self.agreement_year <= y <= current_year])

        if len(years) < 2:
            return 0.0

        h100_years = 0.0

        # Integrate operational H100e over time intervals
        for i in range(len(years) - 1):
            year = years[i]
            next_year = years[i + 1]
            time_increment = next_year - year

            # Get operational H100e at this time (limited by datacenter capacity)
            operational_compute = self.operational_black_project(year)
            h100e_at_year = operational_compute.total_h100e_tpp()

            # Add contribution: H100e * time_increment (in years)
            h100_years += h100e_at_year * time_increment

        return h100_years

    def get_lr_initial(self) -> float:
        """Get initial stock likelihood ratio (constant over time)."""
        lr_prc = self.black_project_stock.lr_from_prc_compute_accounting
        return lr_prc

    def get_lr_sme(self) -> float:
        """Get SME likelihood ratio from covert fab (constant over time). Returns 1.0 if no fab."""
        if self.covert_fab is not None:
            lr_inventory = self.covert_fab.lr_inventory
            lr_procurement = self.covert_fab.lr_procurement
            return lr_inventory * lr_procurement
        return 1.0

    def get_lr_other(self, year: float) -> float:
        years_since_agreement_start = year - self.agreement_year
        return self.lr_over_time_vs_num_workers.get(years_since_agreement_start, 1.0)

    def get_cumulative_evidence_of_covert_project(self, year: float) -> float:
        """
        Calculate cumulative likelihood ratios for all evidence sources at a given time.
        """
        return self.get_lr_initial() * self.get_lr_sme() * self.get_lr_other(year)

    # Fab methods (called by the app)
    def get_fab_lr_inventory(self, year: float) -> float:
        """Get fab inventory detection likelihood ratio. Returns 1.0 if no fab."""
        if self.covert_fab is not None:
            return self.covert_fab.lr_inventory
        return 1.0

    def get_fab_lr_procurement(self, year: float) -> float:
        """Get fab procurement detection likelihood ratio. Returns 1.0 if no fab."""
        if self.covert_fab is not None:
            return self.covert_fab.lr_procurement
        return 1.0

    def get_fab_lr_other(self, year: float) -> float:
        """Get fab other detection likelihood ratio. Returns 1.0 if no fab."""
        if self.covert_fab is not None:
            # Call cumulative_detection_likelihood_ratio to populate lr_other_over_time
            if year not in self.covert_fab.lr_other_over_time:
                self.covert_fab.cumulative_detection_likelihood_ratio(year)
            return self.covert_fab.lr_other_over_time.get(year, 1.0)
        return 1.0

    def get_fab_is_operational(self, year: float) -> float:
        """Get whether fab is operational (1.0 or 0.0). Returns 0.0 if no fab."""
        if self.covert_fab is not None:
            return 1.0 if self.covert_fab.is_operational(year) else 0.0
        return 0.0

    def get_fab_wafer_starts_per_month(self) -> float:
        """Get fab wafer starts per month. Returns 0.0 if no fab."""
        if self.covert_fab is not None:
            return self.covert_fab.wafer_starts_per_month
        return 0.0

    def get_fab_h100_sized_chips_per_wafer(self) -> float:
        """Get fab H100-sized chips per wafer. Returns 0.0 if no fab."""
        if self.covert_fab is not None:
            return self.covert_fab.h100_sized_chips_per_wafer
        return 0.0

    def get_fab_transistor_density_relative_to_h100(self) -> float:
        """Get fab transistor density relative to H100. Returns 1.0 if no fab."""
        if self.covert_fab is not None:
            return self.covert_fab.transistor_density_relative_to_h100
        return 1.0

    def get_fab_process_node(self) -> str:
        """Get fab process node label. Returns empty string if no fab."""
        if self.covert_fab is not None:
            return self.covert_fab.process_node.value
        return ""
