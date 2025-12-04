from typing import Optional
from black_project_backend.classes.black_fab import BlackFab, PRCBlackFab, ProcessNode, FabNotBuiltException, Compute
from black_project_backend.classes.black_project_stock import PRCBlackProjectStock
from black_project_backend.classes.black_datacenters import PRCBlackDatacenters
from black_project_backend.util import lr_over_time_vs_num_workers
from black_project_backend.black_project_parameters import BlackProjectProperties, BlackProjectParameters


class PRCBlackProject:
    """
    A covert PRC infrastructure with compute infrastructure including fabs, datacenters, and dark compute stock.
    """

    def __init__(
        self,
        name: str,
        black_project_properties: BlackProjectProperties,
        agreement_year: float,
        years: list,
        black_project_parameters: BlackProjectParameters,
        stock: Optional[PRCBlackProjectStock] = None,
        black_fab: Optional[BlackFab] = None,
        detection_time_via_other_strategies: Optional[float] = None,
        set_parameters_to_medians: bool = False,
    ):
        self.name = name
        self.black_project_properties = black_project_properties
        self.agreement_year = agreement_year
        self.years = years
        self.black_project_parameters = black_project_parameters
        self.stock = stock
        self.black_fab = black_fab
        self.detection_time_via_other_strategies = detection_time_via_other_strategies
        self.set_parameters_to_medians = set_parameters_to_medians

        # Initialize covert infrastructure
        self._init_covert_infrastructure()

    def _init_covert_infrastructure(self):
        """Initialize all covert infrastructure components."""
        self.black_project_stock = PRCBlackProjectStock(
            agreement_year=self.agreement_year,
            project_parameters=self.black_project_parameters,
            black_project_properties=self.black_project_properties
        )

        # Convert absolute years to years since agreement start
        years_since_agreement_start = [year - self.agreement_year for year in self.years]

        # Calculate energy consumption of PRC stock at agreement start
        # This is used to compute the unconcealed datacenter capacity that can be diverted
        energy_consumption_of_prc_stock_gw = self.black_project_stock.get_energy_consumption_of_prc_stock_gw()

        self.black_datacenters = PRCBlackDatacenters(
            years_since_agreement_start=years_since_agreement_start,
            project_parameters=self.black_project_parameters,
            black_project_properties=self.black_project_properties,
            energy_consumption_of_prc_stock_at_agreement_start=energy_consumption_of_prc_stock_gw,
            agreement_year=self.agreement_year
        )

        if self.black_project_properties.build_a_black_fab:
            try:
                self.black_fab = PRCBlackFab(
                    agreement_year=self.agreement_year,
                    years_since_agreement_start=years_since_agreement_start,
                    project_parameters=self.black_project_parameters,
                    black_project_properties=self.black_project_properties
                )
            except FabNotBuiltException:
                # Fab not built because process node threshold not met
                self.black_fab = None

        # Precompute lr_over_time_vs_num_workers for the total project
        labor_by_year = {}
        for year in self.years:
            # Calculate total labor at this specific year (using absolute year)
            labor_at_year = self.black_datacenters.construction_labor
            labor_at_year += self.black_datacenters.get_operating_labor(year)
            # Add fab labor if it exists
            if self.black_fab is not None:
                labor_at_year += self.black_project_properties.black_fab_construction_labor
                labor_at_year += self.black_project_properties.black_fab_operating_labor
            # Add researcher headcount
            labor_at_year += self.black_project_properties.researcher_headcount
            labor_by_year[year] = int(labor_at_year)

        self.lr_over_time_vs_num_workers = lr_over_time_vs_num_workers(
            labor_by_year=labor_by_year,
            mean_detection_time_100_workers=self.black_project_parameters.detection_parameters.mean_detection_time_for_100_workers,
            mean_detection_time_1000_workers=self.black_project_parameters.detection_parameters.mean_detection_time_for_1000_workers,
            variance_theta=self.black_project_parameters.detection_parameters.variance_of_detection_time_given_num_workers
        )

    def get_operational_compute(self, year: float):
        """Calculate operational compute limited by datacenter energy capacity.

        This gets all surviving compute and scales it down if the energy requirements
        exceed the available datacenter capacity.

        Args:
            year: The year to calculate operational compute for

        Returns:
            Compute object containing chips that can be powered with available capacity
        """
        # Get total datacenter capacity from black_datacenters (concealed + unconcealed)
        datacenter_capacity_gw = self.black_datacenters.get_covert_GW_capacity_total(year)

        # Get all surviving dark compute (using absolute year, not year_since_agreement)
        all_black_project = self.black_project_stock.surviving_compute(year)

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
        using the get_operational_compute method to get capacity-limited compute at each timestep.

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
            operational_compute = self.get_operational_compute(year)
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
        if self.black_fab is not None:
            lr_inventory = self.black_fab.lr_inventory
            lr_procurement = self.black_fab.lr_procurement
            return lr_inventory * lr_procurement
        return 1.0

    def get_lr_other(self, year: float) -> float:
        """Get likelihood ratio from other sources (worker detection).

        Args:
            year: Absolute year (e.g., 2030, 2031, etc.)
        """
        return self.lr_over_time_vs_num_workers.get(year, 1.0)

    def get_cumulative_evidence_of_black_project(self, year: float) -> float:
        """
        Calculate cumulative likelihood ratios for all evidence sources at a given time.

        Args:
            year: Absolute year (e.g., 2030, 2031, etc.)
        """
        satellite_lr = self.black_datacenters.lr_from_identifying_datacenters_with_satellites()
        energy_lr = self.black_datacenters.lr_from_reported_energy_consumption(year)
        base_lr = self.get_lr_initial() * self.get_lr_sme() * satellite_lr * energy_lr * self.get_lr_other(year)
        return base_lr

    # Fab methods (called by the app)
    def get_fab_lr_inventory(self, year: float) -> float:
        """Get fab inventory detection likelihood ratio. Returns 1.0 if no fab."""
        if self.black_fab is not None:
            return self.black_fab.lr_inventory
        return 1.0

    def get_fab_lr_procurement(self, year: float) -> float:
        """Get fab procurement detection likelihood ratio. Returns 1.0 if no fab."""
        if self.black_fab is not None:
            return self.black_fab.lr_procurement
        return 1.0

    def get_fab_lr_other(self, year: float) -> float:
        """Get fab other detection likelihood ratio. Returns 1.0 if no fab."""
        if self.black_fab is not None:
            # Call cumulative_detection_likelihood_ratio to populate lr_other_over_time
            if year not in self.black_fab.lr_other_over_time:
                self.black_fab.cumulative_detection_likelihood_ratio(year)
            return self.black_fab.lr_other_over_time.get(year, 1.0)
        return 1.0

    def get_fab_is_operational(self, year: float) -> float:
        """Get whether fab is operational (1.0 or 0.0). Returns 0.0 if no fab."""
        if self.black_fab is not None:
            return 1.0 if self.black_fab.is_operational(year) else 0.0
        return 0.0

    def get_fab_wafer_starts_per_month(self) -> float:
        """Get fab wafer starts per month. Returns 0.0 if no fab."""
        if self.black_fab is not None:
            return self.black_fab.wafer_starts_per_month
        return 0.0

    def get_fab_h100_sized_chips_per_wafer(self) -> float:
        """Get fab H100-sized chips per wafer. Returns 0.0 if no fab."""
        if self.black_fab is not None:
            return self.black_fab.h100_sized_chips_per_wafer
        return 0.0

    def get_fab_transistor_density_relative_to_h100(self) -> float:
        """Get fab transistor density relative to H100. Returns 1.0 if no fab."""
        if self.black_fab is not None:
            return self.black_fab.transistor_density_relative_to_h100
        return 1.0

    def get_fab_process_node(self) -> str:
        """Get fab process node label. Returns empty string if no fab."""
        if self.black_fab is not None:
            return self.black_fab.process_node.value
        return ""
