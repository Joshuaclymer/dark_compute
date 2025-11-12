from dataclasses import dataclass
from abc import ABC, abstractmethod
from util import sample_from_log_normal

@dataclass
class CovertDatacenterParameters():
    # -- Energy capacity --
    max_proportion_of_PRC_energy_consumption = 0.05
    total_GW_of_PRC_energy_consumption = 1100

    typical_construction_workers_per_GW = 1000
    relative_sigma_typical_construction_workers_per_GW = 0.1

    # -- Detection evidence --

    # Data centers built for concealment
    operating_labor_per_GW = 100
    relative_sigma_operating_labor_per_gw = 0.1

    mean_detection_time_of_covert_site_for_100_workers = 6.95   # Small operation: ~7 years to detect
    mean_detection_time_of_covert_site_for_1000_workers = 3.42  # Medium operation: ~3.4 years to detect
    variance_of_detection_time_given_num_workers = 3.880

    # Data centers not built for concealment

    meters_squared_of_footprint_area_per_gw = 1e-2
    relative_sigma_meters_squared_of_footprint_per_gw = 0.1

def sample_operating_labor(energy_capacity):
    median = CovertDatacenterParameters.operating_labor_per_gw * energy_capacity
    relative_sigma = CovertDatacenterParameters.relative_sigma_operating_labor_per_gw
    return sample_from_log_normal(median, relative_sigma)

# def sample_GW_constructed_per_year(construction_labor):


class CovertDataCenters(ABC):
    number_of_initially_concealed_datacenters : float 
    GW_per_initially_concealed_datacenter : float
    construction_labor : float
    operating_labor : float
    GW_constructed_per_year : float
    
    @abstractmethod
    def get_energy_capacity(self, year : float) -> float:
        return max(self.GW_constructed_per_year * year,
            self.number_of_initially_concealed_datacenters * self.GW_per_initially_concealed_datacenter)

@dataclass
class CovertPRCDatacenters(CovertDataCenters):
    number_of_initially_concealed_datacenters : float 
    GW_per_initially_concealed_datacenter : float
    construction_labor : float
    operating_labor : float

    def __init__(self, number_of_initially_concealed_datacenters : int, GW_per_initially_concealed_datacenter : float, construction_labor: int):
        self.number_of_initially_concealed_datacenters = number_of_initially_concealed_datacenters 
        self.GW_per_initially_concealed_datacenter = GW_per_initially_concealed_datacenter
        self.construction_labor = construction_labor
        self.operating_labor = sample_operating_labor(self.energy_capacity)
    




