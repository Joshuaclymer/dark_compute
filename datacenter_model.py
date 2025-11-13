from dataclasses import dataclass
from abc import ABC, abstractmethod
from util import sample_from_log_normal
import numpy as np
from scipy import stats

@dataclass
class CovertDatacenterParameters():

    # Data centers built for concealment

    # -- Energy capacity --
    max_proportion_of_PRC_energy_consumption = 0.05
    total_GW_of_PRC_energy_consumption = 1100

    # -- Detection evidence --
    construction_labor_per_MW_per_year = 6.3
    relative_sigma_construction_labor_per_MW_per_year = 0.4

    operating_labor_per_MW = 1
    relative_sigma_operating_labor_per_MW = 0.4

    mean_detection_time_of_covert_site_for_100_workers = 6.95   # Small operation: ~7 years to detect
    mean_detection_time_of_covert_site_for_1000_workers = 3.42  # Medium operation: ~3.4 years to detect
    variance_of_detection_time_given_num_workers = 3.880

    # Data centers not built for concealment

    # meters_squared_of_footprint_area_per_gw = 1e-2
    # relative_sigma_meters_squared_of_footprint_per_gw = 0.1

def sample_construction_labor_per_GW_per_year():
    median = CovertDatacenterParameters.construction_labor_per_MW_per_year * 1000
    relative_sigma = CovertDatacenterParameters.relative_sigma_construction_labor_per_MW_per_year
    return sample_from_log_normal(median, relative_sigma)

def sample_operating_labor_per_GW():
    median = CovertDatacenterParameters.operating_labor_per_MW * 1000
    relative_sigma = CovertDatacenterParameters.relative_sigma_operating_labor_per_MW
    return sample_from_log_normal(median, relative_sigma)

# Pre-compute constants for detection time calculations
_detection_time_A = None
_detection_time_B = None
_detection_time_theta = None

def _initialize_detection_constants():
    """Pre-compute constants used in detection time calculations."""
    global _detection_time_A, _detection_time_B, _detection_time_theta

    if _detection_time_A is not None:
        return  # Already initialized

    x1 = 100
    mu1 = CovertDatacenterParameters.mean_detection_time_of_covert_site_for_100_workers
    x2 = 1000
    mu2 = CovertDatacenterParameters.mean_detection_time_of_covert_site_for_1000_workers

    _detection_time_B = np.log(mu1 / mu2) / np.log(np.log10(x2) / np.log10(x1))
    _detection_time_A = mu1 * (np.log10(x1) ** _detection_time_B)
    _detection_time_theta = CovertDatacenterParameters.variance_of_detection_time_given_num_workers

def clear_detection_constants_cache():
    """Clear cached detection constants. Call this when FabModelParameters change."""
    global _detection_time_A, _detection_time_B, _detection_time_theta
    _detection_time_A = None
    _detection_time_B = None
    _detection_time_theta = None

def sample_time_of_detection_via_other_strategies(total_labor: int) -> float:
    """
    Sample the time when detection occurs via other intelligence strategies (HUMINT, SIGINT, etc.).

    Based on historical nuclear program detection data. Detection timing follows a Gamma distribution
    whose mean depends on the number of workers involved.

    Args:
        total_labor: Total number of workers involved (construction + operation)

    Returns:
        float: Years from construction start until detection occurs
    """
    if total_labor <= 0:
        return float('inf')  # Never detected if no workers

    # Initialize constants if not already done
    _initialize_detection_constants()

    # Mean detection time: μ(x) = A / log10(x)^B
    mu = _detection_time_A / (np.log10(total_labor) ** _detection_time_B)

    # Gamma distribution parameters: mean = k*θ, variance = k*θ²
    # Given: mean = μ, variance = σ²*μ (proportional to mean)
    # Solving: θ = σ², k = μ/σ²
    k = mu / _detection_time_theta

    # Sample from Gamma distribution
    detection_time = stats.gamma.rvs(a=k, scale=_detection_time_theta)

    return detection_time


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

        # Calculate total labor and sample detection time once
        # Use a representative time point (e.g., year 5) for labor calculation
        representative_capacity = self.GW_per_initial_datacenter * self.number_of_initial_datacenters + self.GW_per_year_of_concealed_datacenters * 5
        construction_labor = self.construction_labor_per_GW_per_year * self.GW_per_year_of_concealed_datacenters
        operating_labor = self.operating_labor_per_GW * representative_capacity
        total_labor = construction_labor + operating_labor
        self.time_of_detection = sample_time_of_detection_via_other_strategies(total_labor)

        # Cache for operating labor calculation
        self.construction_labor = 0
        self.operating_labor = 0
    
    def get_GW_capacity(self, year):
        return self.GW_per_initial_datacenter * self.number_of_initial_datacenters + self.GW_per_year_of_concealed_datacenters * year

    def get_operating_labor(self, year):
        self.operating_labor = self.operating_labor_per_GW * self.get_GW_capacity(year)
        return self.operating_labor

    def lr_from_concealed_datacenters(self, year):
        # Use pre-computed labor parameters and detection time
        construction_labor = self.construction_labor_per_GW_per_year * self.GW_per_year_of_concealed_datacenters
        operating_labor = self.get_operating_labor(year)
        total_labor = construction_labor + operating_labor
        return lr_from_other_strategies(year, self.time_of_detection, total_labor)