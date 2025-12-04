import numpy as np
from scipy import stats
from dataclasses import dataclass
from abc import ABC, abstractmethod
from black_project_backend.util import sample_from_log_normal, lr_over_time_vs_num_workers, sample_us_estimate_with_error, lr_from_discrepancy_in_us_estimate
from black_project_backend.black_project_parameters import BlackDatacenterParameters, BlackProjectParameters, BlackProjectProperties
from typing import List, Dict

def sample_GW_per_year_per_construction_labor(datacenter_parameters):
    median = datacenter_parameters.MW_per_construction_worker_per_year / 1000
    relative_sigma = datacenter_parameters.relative_sigma_mw_per_construction_worker_per_year
    return sample_from_log_normal(median, relative_sigma)

class PRCBlackDatacenters():
    def __init__(
        self,
        years_since_agreement_start: List[float],
        project_parameters: BlackProjectParameters,
        black_project_properties: BlackProjectProperties,
        energy_consumption_of_prc_stock_at_agreement_start: float,
        agreement_year: int = 2030
    ):
        # Store the main parameters object
        self.project_parameters = project_parameters
        self.black_project_properties = black_project_properties

        # Extract commonly used sub-parameters for convenience
        self.datacenter_parameters = project_parameters.datacenter_model_parameters

        # Get values from black_project_properties
        self.construction_labor = black_project_properties.datacenter_construction_labor
        years_before = black_project_properties.years_before_agreement_year_prc_starts_building_black_datacenters
        self.datacenter_start_year_offset = -years_before
        self.energy_consumption_of_prc_stock_at_agreement_start = energy_consumption_of_prc_stock_at_agreement_start

        # Sample labor parameters once at initialization for efficiency
        self.GW_per_year_per_construction_labor = sample_GW_per_year_per_construction_labor(self.datacenter_parameters)
        self.GW_per_year_of_concealed_datacenters = self.GW_per_year_per_construction_labor * self.construction_labor
        self.operating_labor_per_GW = self.datacenter_parameters.operating_labor_per_MW * 1000

        # Store agreement year for later use
        self.agreement_year = agreement_year

        # Convert relative years to absolute years for labor calculation
        absolute_years = [year + agreement_year for year in years_since_agreement_start]

        # Build labor_by_year using relative years as keys (for lr_over_time_vs_num_workers)
        # but use absolute years to calculate labor (for get_operating_labor)
        labor_by_relative_year = {
            relative_year: self.get_operating_labor(relative_year + agreement_year) + self.construction_labor
            for relative_year in years_since_agreement_start
        }
        self.lr_over_time_vs_num_workers = lr_over_time_vs_num_workers(
            labor_by_year=labor_by_relative_year,
            mean_detection_time_100_workers=project_parameters.detection_parameters.mean_detection_time_for_100_workers,
            mean_detection_time_1000_workers=project_parameters.detection_parameters.mean_detection_time_for_1000_workers,
            variance_theta=project_parameters.detection_parameters.variance_of_detection_time_given_num_workers
        )

    def get_covert_GW_capacity_unconcealed_at_agreement_start(self):
        fraction_diverted = self.black_project_properties.fraction_of_datacenter_capacity_not_built_for_concealment_diverted_to_black_project_at_agreement_start
        return fraction_diverted * self.energy_consumption_of_prc_stock_at_agreement_start

    def get_covert_GW_capacity_concealed(self, year):
        """Calculate the GW capacity of datacenters built for concealment.

        Args:
            year: Absolute year (e.g., 2030, 2031, etc.)
        """
        # Convert absolute year to year relative to agreement start
        year_relative_to_agreement = year - self.agreement_year

        # Adjust by datacenter start offset (only count years after construction starts)
        years_since_construction_start = max(0, year_relative_to_agreement - self.datacenter_start_year_offset)
        constructable_datacenters = self.GW_per_year_of_concealed_datacenters * years_since_construction_start
        max_energy_allocatable = self.black_project_properties.max_proportion_of_PRC_energy_consumption * self.project_parameters.exogenous_trends.total_GW_of_PRC_energy_consumption
        maximum_constructable_while_keeping_within_energy_limit = max(0, max_energy_allocatable - self.get_covert_GW_capacity_unconcealed_at_agreement_start())
        constructed_covert_datacenters = min(constructable_datacenters, maximum_constructable_while_keeping_within_energy_limit)
        return constructed_covert_datacenters


    def get_covert_GW_capacity_total(self, year):
        """Calculate the total GW capacity of covert datacenters (concealed + unconcealed)."""
        built_for_concealment = self.get_covert_GW_capacity_concealed(year)
        not_built_for_concealment = self.get_covert_GW_capacity_unconcealed_at_agreement_start()
        return built_for_concealment + not_built_for_concealment

    def get_operating_labor(self, year):
        return self.operating_labor_per_GW * self.get_covert_GW_capacity_total(year)

    def cumulative_lr_from_direct_observation(self, year):
        """Get the cumulative likelihood ratio from concealed datacenters based on worker count.

        Args:
            year: Absolute year (e.g., 2030, 2031, etc.)
        """
        relative_year = year - self.agreement_year
        return self.lr_over_time_vs_num_workers.get(relative_year, 1.0)

    def lr_from_reported_energy_consumption(self, year: float) -> float:
        """Calculate likelihood ratio from PRC energy consumption accounting.

        Compares:
        - Case 1: Black project exists - covert datacenters consume energy that creates
          a discrepancy between total PRC energy and what's accounted for
        - Case 2: No black project - all energy consumption is accounted for

        Args:
            year: The year to calculate the likelihood ratio for

        Returns:
            Likelihood ratio (probability of observing US estimate if project exists /
            probability if project doesn't exist)
        """
        total_prc_energy = self.project_parameters.exogenous_trends.total_GW_of_PRC_energy_consumption
        covert_energy = self.get_covert_GW_capacity_total(year)
        median_error = self.project_parameters.detection_parameters.us_intelligence_median_error_in_energy_consumption_estimate_of_prc_datacenter_capacity

        # Handle edge case where there's no covert energy
        if covert_energy < 1e-10 or total_prc_energy < 1e-10:
            return 1.0  # Neutral evidence if no energy consumption

        # Energy consumed by everything except covert datacenters (what PRC reports)
        reported_energy = total_prc_energy - covert_energy
        if reported_energy < 1e-10:
            return 1.0  # Edge case: if reported energy is ~0, return neutral

        # Sample US estimate of total PRC energy consumption
        us_estimate = sample_us_estimate_with_error(total_prc_energy, median_error)

        return lr_from_discrepancy_in_us_estimate(
            true_if_project_exists=total_prc_energy,
            true_if_no_project=reported_energy,
            us_estimate=us_estimate,
            median_error=median_error
        )

    def lr_from_identifying_datacenters_with_satellites(self) -> float:
        """Calculate likelihood ratio from satellite detection of diverted datacenter capacity.

        The US uses satellite imagery to estimate the capacity of unconcealed datacenters.
        The PRC reports capacity that excludes the diverted portion.

        Compares:
        - Case 1: Black project exists - some datacenter capacity was diverted, so the
          true capacity of unconcealed datacenters equals what PRC reports
        - Case 2: No black project - no capacity was diverted, so true capacity equals
          the full energy_consumption_of_prc_stock_at_agreement_start

        Returns:
            Likelihood ratio (probability of observing US estimate if project exists /
            probability if project doesn't exist)
        """
        diverted_capacity = self.get_covert_GW_capacity_unconcealed_at_agreement_start()
        median_error = self.project_parameters.detection_parameters.us_intelligence_median_error_in_satellite_estimate_of_prc_datacenter_capacity

        # Handle edge case where there's no diverted capacity
        if diverted_capacity < 1e-10:
            return 1.0  # Neutral evidence if no diverted capacity

        # Total datacenter capacity before diversion
        total_capacity = self.energy_consumption_of_prc_stock_at_agreement_start
        if total_capacity < 1e-10:
            return 1.0  # Edge case: no datacenter capacity

        # Reported capacity (what PRC claims - excludes diverted portion)
        reported_capacity = total_capacity - diverted_capacity
        if reported_capacity < 1e-10:
            return 1.0  # Edge case

        # Sample US satellite estimate of unconcealed datacenter capacity
        us_estimate = sample_us_estimate_with_error(reported_capacity, median_error)

        return lr_from_discrepancy_in_us_estimate(
            true_if_project_exists=reported_capacity,
            true_if_no_project=total_capacity,
            us_estimate=us_estimate,
            median_error=median_error
        )

    def cumulative_lr_from_concealed_datacenters(self, year):
        """Get the cumulative likelihood ratio from concealed datacenters based on worker count."""
        return self.lr_from_identifying_datacenters_with_satellites() * self.lr_from_reported_energy_consumption(year) * self.cumulative_lr_from_direct_observation(year)
