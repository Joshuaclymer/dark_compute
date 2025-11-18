import numpy as np
from scipy import stats
from dataclasses import dataclass
from abc import ABC, abstractmethod
from backend.util import sample_from_log_normal, lr_vs_num_workers
from backend.paramaters import CovertDatacenterParameters, CovertProjectParameters

def sample_construction_labor_per_GW_per_year():
    median = CovertDatacenterParameters.construction_labor_per_MW_per_year * 1000
    relative_sigma = CovertDatacenterParameters.relative_sigma_construction_labor_per_MW_per_year
    return sample_from_log_normal(median, relative_sigma)

def sample_operating_labor_per_GW():
    median = CovertDatacenterParameters.operating_labor_per_MW * 1000
    relative_sigma = CovertDatacenterParameters.relative_sigma_operating_labor_per_MW
    return sample_from_log_normal(median, relative_sigma)


def lr_from_other_strategies(
    years_since_agreement_start: float,
    time_of_detection: float,
    total_labor: int
) -> float:
    """
    Likelihood ratio from other intelligence strategies (HUMINT, SIGINT, etc.).

    Uses the actual Gamma distribution of detection times to calculate principled likelihood ratio.

    Evidence is whether detection has occurred by current time:
    - If detected: LR is very high (strong evidence for fab)
    - If not detected: LR = P(not detected by time t | fab exists) / P(not detected | no fab)
                          = P(detection time ≥ t | fab exists) / 1.0
                          = Survival function of Gamma distribution at t
                          = 1 - CDF_Gamma(t)

    Args:
        years_since_construction_start: Years elapsed since construction began
        time_of_detection: Sampled time (in years from start) when detection occurs
        total_labor: Total number of workers (needed to get Gamma distribution parameters)

    Returns:
        float: Likelihood ratio LR = P(evidence|fab exists) / P(evidence|fab doesn't exist)
    """
    if years_since_agreement_start < 0:
        return 1.0  # Project hasn't started yet

    # Check if current time has passed the detection time
    if years_since_agreement_start >= time_of_detection:
        return 100  # Very high LR - project has been detected
    else:
        # Evidence: No detection yet despite time elapsed
        # Need to calculate P(detection time ≥ t_current | fab exists)

        if total_labor <= 0:
            return 1.0  # No workers, no detection possible

        # Initialize constants if not already done
        _initialize_detection_constants()

        # Mean detection time: μ(x) = A / log10(x)^B
        mu = _detection_time_A / (np.log10(total_labor) ** _detection_time_B)

        # Gamma distribution parameters
        k = mu / _detection_time_theta

        # P(not detected by time t | fab exists) = P(T ≥ t) = 1 - CDF(t)
        # This is the survival function (sf) of the Gamma distribution
        p_not_detected_given_fab = stats.gamma.sf(years_since_agreement_start, a=k, scale=_detection_time_theta)

        # P(not detected | no fab) = 1.0 (no fab means never detected)
        p_not_detected_given_no_fab = 1.0

        # Likelihood ratio
        lr = p_not_detected_given_fab / p_not_detected_given_no_fab

        return max(lr, 0.001)  # Floor at 0.001 to avoid numerical issues
    


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
    def __init__(self, GW_per_initial_datacenter : float, number_of_initial_datacenters : float, GW_per_year_of_concealed_datacenters : float):
        self.GW_per_initial_datacenter = GW_per_initial_datacenter
        self.number_of_initial_datacenters = number_of_initial_datacenters
        self.GW_per_year_of_concealed_datacenters = GW_per_year_of_concealed_datacenters

        # Sample labor parameters once at initialization for efficiency
        self.construction_labor_per_GW_per_year = sample_construction_labor_per_GW_per_year()
        self.operating_labor_per_GW = sample_operating_labor_per_GW()

    def get_GW_capacity(self, year):
        return self.GW_per_initial_datacenter * self.number_of_initial_datacenters + self.GW_per_year_of_concealed_datacenters * year

    def get_operating_labor(self, year):
        return self.operating_labor_per_GW * self.get_GW_capacity(year)
    
    def get_construction_labor(self):
        return self.construction_labor_per_GW_per_year * self.GW_per_year_of_concealed_datacenters

    def lr_from_concealed_datacenters(self, year):
        # Use pre-computed labor parameters and detection time
        construction_labor = self.get_construction_labor()
        operating_labor = self.get_operating_labor(year)
        total_labor = construction_labor + operating_labor
        return lr_vs_num_workers(
            years_since_start=year,
            total_labor=total_labor,
            mean_detection_time_100_workers=CovertProjectParameters.mean_detection_time_for_100_workers,
            mean_detection_time_1000_workers=CovertProjectParameters.mean_detection_time_for_1000_workers,
            variance_theta=CovertProjectParameters.variance_of_detection_time_given_num_workers
        )
