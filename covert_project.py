from dataclasses import dataclass
from typing import Optional
from fab_model import CovertFab, PRCCovertFab, ProcessNode, FabNotBuiltException, Compute
from stock_model import PRCDarkComputeStock
from datacenter_model import CovertPRCDatacenters

@dataclass
class CovertProjectStrategy:
    run_a_covert_project : bool
    # Initial compute stock
    proportion_of_initial_compute_stock_to_divert : Optional[float] = None

    # Data centers
    GW_per_initial_datacenter : float = 5
    number_of_initial_datacenters : float = 0.1
    GW_per_year_of_concealed_datacenters : float = 1

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
    stock : Optional[PRCDarkComputeStock] = None
    covert_fab : Optional[CovertFab] = None

    def __post_init__(self):
        # Import here to avoid circular dependency
        from model import best_prc_covert_project_strategy

        # Initialize covert fab if strategy requires it and construction_start_year is set
        self.dark_compute_stock = PRCDarkComputeStock(
            agreement_year = self.agreement_year,
            proportion_of_initial_compute_stock_to_divert = self.covert_project_strategy.proportion_of_initial_compute_stock_to_divert,
            optimal_proportion_of_initial_compute_stock_to_divert = best_prc_covert_project_strategy.proportion_of_initial_compute_stock_to_divert
        )

        self.covert_datacenters = CovertPRCDatacenters(
            GW_per_initial_datacenter = self.covert_project_strategy.GW_per_initial_datacenter,
            number_of_initial_datacenters = self.covert_project_strategy.number_of_initial_datacenters,
            GW_per_year_of_concealed_datacenters = self.covert_project_strategy.GW_per_year_of_concealed_datacenters
        )

        if (self.covert_project_strategy.build_a_covert_fab):
            try:
                self.covert_fab = PRCCovertFab(
                    construction_start_year = self.agreement_year,
                    construction_labor = self.covert_project_strategy.covert_fab_construction_labor,
                    process_node = self.covert_project_strategy.covert_fab_process_node,
                    proportion_of_prc_lithography_scanners_devoted_to_fab = self.covert_project_strategy.covert_fab_proportion_of_prc_lithography_scanners_devoted,
                    operation_labor = self.covert_project_strategy.covert_fab_operating_labor,
                    agreement_year = self.agreement_year
                )
            except FabNotBuiltException:
                # Fab not built because process node threshold not met
                self.covert_fab = None

    def operational_dark_compute(self, year: float):
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
        all_dark_compute = self.dark_compute_stock.dark_compute(year)

        # Calculate total energy requirements
        total_energy_required_gw = all_dark_compute.total_energy_requirements_GW()

        # If energy requirements are within capacity, return all dark compute
        if total_energy_required_gw <= datacenter_capacity_gw:
            return all_dark_compute

        # Otherwise, scale down chip counts proportionally to fit within capacity
        scaling_factor = datacenter_capacity_gw / total_energy_required_gw

        # Create new chip_counts with scaled values
        operational_chip_counts = {
            chip: count * scaling_factor
            for chip, count in all_dark_compute.chip_counts.items()
        }

        operational_compute = Compute(chip_counts=operational_chip_counts)

        print(f"DEBUG operational_dark_compute(year={year}): capacity={datacenter_capacity_gw:.2f}GW, "
              f"required={total_energy_required_gw:.2f}GW, scaling_factor={scaling_factor:.4f}, "
              f"operational_tpp={operational_compute.total_h100e_tpp():.2f}", flush=True)

        return operational_compute

    def h100_years_before_detection(self, years: list[float], detection_year: float = None) -> float:
        """Calculate total H100-years of computation before detection.

        This integrates operational H100e over time from agreement start until detection
        (or end of simulation if never detected).

        Args:
            years: List of simulation years
            detection_year: Year when project is detected (None if never detected)

        Returns:
            Total H100-years of computation (H100e * years)
        """
        # Filter years up to detection (or all years if never detected)
        if detection_year is not None:
            years_to_integrate = [y for y in years if self.agreement_year <= y <= detection_year]
        else:
            years_to_integrate = [y for y in years if self.agreement_year <= y]

        h100_years = 0.0

        # Integrate operational H100e over time intervals
        for i in range(len(years_to_integrate) - 1):
            year = years_to_integrate[i]
            next_year = years_to_integrate[i + 1]
            time_increment = next_year - year

            # Get operational H100e at this time using our operational_dark_compute method
            operational_at_year = self.operational_dark_compute(year)
            h100e_at_year = operational_at_year.total_h100e_tpp()

            # Add contribution: H100e * time_increment (in years)
            h100_years += h100e_at_year * time_increment

        return h100_years

    def get_aggregated_likelihood_ratios(self, year: float) -> dict:
        """Calculate aggregated likelihood ratios for all evidence sources at a given time.

        This calculates the likelihood ratio in the same way as the frontend
        "Posterior Probability of Covert Project" calculation:

        project_lr = lr_initial × lr_sme × lr_other(t)

        Where:
        - lr_initial = lr_prc × lr_global (from initial compute stock accounting, constant)
        - lr_sme = lr_inventory × lr_procurement (from fab SME intelligence, constant)
        - lr_other = lr_other(t) (from fab other intelligence, varies over time)

        Note: Inventory and procurement LRs are constant throughout the simulation because
        they depend only on fab construction parameters (scanner counts, localization timing),
        not on time. Only lr_other varies with time.

        Returns a dictionary with:
        - 'project_lr': Combined likelihood ratio for project existence
        - 'fab_lr': Combined likelihood ratio for fab existence (if fab exists)
        - 'sources': Dictionary mapping source names to individual LRs

        Args:
            year: The year to calculate likelihood ratios for

        Returns:
            dict with 'project_lr', 'fab_lr', and 'sources' keys
        """
        from model import UpdateSource

        sources = {}

        # Get likelihood ratios from dark compute stock (initial stock evidence)
        # These are constant per simulation
        lr_prc = self.dark_compute_stock.lr_from_prc_compute_accounting
        lr_global = self.dark_compute_stock.lr_from_global_compute_production_accounting

        sources[UpdateSource.PRC_COMPUTE_ACCOUNTING] = lr_prc
        sources[UpdateSource.GLOBAL_COMPUTE_PRODUCTION_ACCOUNTING] = lr_global

        # Combined initial stock LR (constant)
        lr_initial = lr_prc * lr_global

        # Initialize project LR with initial stock evidence
        project_lr = lr_initial

        # Get likelihood ratios from covert fab if it exists
        fab_lr = 1.0  # Neutral evidence if no fab
        if self.covert_fab is not None:
            # Call detection_likelihood_ratio to populate lr_other for this year
            # This also updates lr_inventory and lr_procurement, but those are constant
            self.covert_fab.detection_likelihood_ratio(year=year)

            # Get individual component LRs
            # lr_inventory and lr_procurement are constant (based on construction parameters)
            # lr_other varies with time (based on detection probability)
            lr_inventory = self.covert_fab.lr_inventory
            lr_procurement = self.covert_fab.lr_procurement
            lr_other = self.covert_fab.lr_other

            sources[UpdateSource.FAB_INVENTORY_INTELLIGENCE] = lr_inventory
            sources[UpdateSource.FAB_PROCUREMENT_INTELLIGENCE] = lr_procurement
            sources[UpdateSource.FAB_OTHER_INTELLIGENCE] = lr_other

            # Calculate SME evidence (inventory × procurement) - constant
            lr_sme = lr_inventory * lr_procurement

            # Combined fab LR
            fab_lr = lr_sme * lr_other

            # Project LR = initial × sme × other (matching frontend calculation)
            # Only lr_other varies with time, so this matches the frontend trend
            project_lr = lr_initial * lr_sme * lr_other

        return {
            'project_lr': project_lr,
            'fab_lr': fab_lr,
            'sources': sources
        }

    def h100_years_to_date(self, current_year: float, years_list: list[float]) -> float:
        """Calculate total H100-years of computation performed up to the current year.

        This integrates operational H100e over time from agreement start until the current year,
        using the operational_dark_compute method to get capacity-limited compute at each timestep.

        Args:
            current_year: The year to calculate H100-years up to
            years_list: List of years to integrate over (should include agreement_year through current_year)

        Returns:
            float: Total H100-years of computation performed (H100e × years)
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
            operational_compute = self.operational_dark_compute(year)
            h100e_at_year = operational_compute.total_h100e_tpp()

            # Add contribution: H100e * time_increment (in years)
            h100_years += h100e_at_year * time_increment

        return h100_years

    def get_fab_lr_inventory(self, year: float) -> float:
        """Get fab inventory detection likelihood ratio. Returns 1.0 if no fab."""
        if self.covert_fab is not None and hasattr(self.covert_fab, 'lr_inventory_over_time') and year in self.covert_fab.lr_inventory_over_time:
            return self.covert_fab.lr_inventory_over_time[year]
        return 1.0

    def get_fab_lr_procurement(self, year: float) -> float:
        """Get fab procurement detection likelihood ratio. Returns 1.0 if no fab."""
        if self.covert_fab is not None and hasattr(self.covert_fab, 'lr_procurement_over_time') and year in self.covert_fab.lr_procurement_over_time:
            return self.covert_fab.lr_procurement_over_time[year]
        return 1.0

    def get_fab_lr_other(self, year: float) -> float:
        """Get fab other detection likelihood ratio. Returns 1.0 if no fab."""
        if self.covert_fab is not None and hasattr(self.covert_fab, 'lr_other_over_time') and year in self.covert_fab.lr_other_over_time:
            return self.covert_fab.lr_other_over_time[year]
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

# ============================================================================
# DEFAULTS

default_prc_covert_project_strategy = CovertProjectStrategy(
    run_a_covert_project = True,

    proportion_of_initial_compute_stock_to_divert = 0.05,

    # Covert fabs
    build_a_covert_fab = True,
    covert_fab_operating_labor = 550,
    covert_fab_construction_labor = 250,
    covert_fab_process_node = "best_indigenous",
    covert_fab_proportion_of_prc_lithography_scanners_devoted = 0.1,
)

best_prc_covert_project_strategy = default_prc_covert_project_strategy
