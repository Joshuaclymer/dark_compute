import numpy as np
from scipy import stats
from dataclasses import dataclass
from abc import ABC, abstractmethod
from backend.util import sample_from_log_normal, lr_over_time_vs_num_workers
from backend.paramaters import CovertDatacenterParameters, CovertProjectParameters
from typing import List, Dict

def sample_GW_per_year_per_construction_labor(datacenter_parameters):
    median = datacenter_parameters.MW_per_construction_worker_per_year / 1000
    relative_sigma = datacenter_parameters.relative_sigma_mw_per_construction_worker_per_year
    return sample_from_log_normal(median, relative_sigma)


# class CovertDataCenters(ABC):
#     number_of_initially_concealed_datacenters : float
#     GW_per_initially_concealed_datacenter : float
#     construction_labor : float
#     operating_labor : float
#     GW_constructed_per_year : float

#     @abstractmethod
#     def get_energy_capacity(self, year : float) -> float:
#         return max(self.GW_constructed_per_year * year,
#             self.number_of_initially_concealed_datacenters * self.GW_per_initially_concealed_datacenter)

class CovertPRCDatacenters():
    def __init__(self, GW_per_initial_datacenter : float, number_of_initial_datacenters : float, construction_labor : float, years_since_agreement_start: List[float], datacenter_parameters: CovertDatacenterParameters, project_parameters: CovertProjectParameters):
        self.GW_per_initial_datacenter = GW_per_initial_datacenter
        self.number_of_initial_datacenters = number_of_initial_datacenters
        self.construction_labor = construction_labor

        # Store datacenter parameters instance
        self.datacenter_parameters = datacenter_parameters
        self.project_parameters = project_parameters

        # Sample labor parameters once at initialization for efficiency
        self.GW_per_year_per_construction_labor = sample_GW_per_year_per_construction_labor(datacenter_parameters)
        self.GW_per_year_of_concealed_datacenters = self.GW_per_year_per_construction_labor * self.construction_labor
        self.operating_labor_per_GW = self.datacenter_parameters.operating_labor_per_MW * 1000
        labor_by_year = {
            year: self.get_operating_labor(year) + self.construction_labor
            for year in years_since_agreement_start
        }
        self.lr_over_time_vs_num_workers = lr_over_time_vs_num_workers(
            labor_by_year=labor_by_year,
            mean_detection_time_100_workers=project_parameters.mean_detection_time_for_100_workers,
            mean_detection_time_1000_workers=project_parameters.mean_detection_time_for_1000_workers,
            variance_theta=project_parameters.variance_of_detection_time_given_num_workers
        )

    def get_GW_capacity(self, year):
        constructable_datacenters = self.GW_per_initial_datacenter * self.number_of_initial_datacenters + self.GW_per_year_of_concealed_datacenters * year
        max_energy_allocatable = self.datacenter_parameters.max_proportion_of_PRC_energy_consumption * self.datacenter_parameters.total_GW_of_PRC_energy_consumption
        return min(constructable_datacenters, max_energy_allocatable)

    def get_operating_labor(self, year):
        return self.operating_labor_per_GW * self.get_GW_capacity(year)

    def cumulative_lr_from_concealed_datacenters(self, year):
        lr = self.lr_over_time_vs_num_workers.get(year, 1.0)
        return lr
