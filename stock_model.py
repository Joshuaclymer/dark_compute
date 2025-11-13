import numpy as np
import random
from typing import List, Dict
from abc import ABC, abstractmethod

class InitialPRCComputeStockParameters():
    
    # PRC compute stock
    total_prc_compute_stock_in_2025 = 1e6
    annual_growth_rate_of_prc_compute_stock = 2.4
    relative_sigma_of_prc_compute_stock = 0.1

    us_intelligence_median_error_in_estimate_of_prc_compute_stock = 0.07

    # Global compute stock parameters
    total_global_compute_in_2025 = 1e7
    annual_growth_rate_of_global_compute = 2.4
    relative_sigma_of_global_compute = 0.1

    median_unreported_compute_owned_by_non_prc_actors = 1e6
    relative_sigma_unreported_compute_owned_by_non_prc_actors = 0.5

class SurvivalRateParameters():

    # Survival rate parameters
    initial_hazard_rate_p50 = 0.01
    increase_of_hazard_rate_per_year_p50 = 0.0035
    hazard_rate_p25_relative_to_p50 = 0.1
    hazard_rate_p75_relative_to_p50 = 6

def sample_initial_prc_compute_stock(year):
    """Sample initial PRC compute stock based on year and proportion diverted to covert project"""
    from util import sample_from_log_normal
    params = InitialPRCComputeStockParameters
    years_since_2025 = year - 2025
    median_total_stock = params.total_prc_compute_stock_in_2025 * (params.annual_growth_rate_of_prc_compute_stock ** years_since_2025)
    relative_sigma = params.relative_sigma_of_prc_compute_stock

    total_stock_sample = sample_from_log_normal(median_total_stock, relative_sigma)

    return total_stock_sample

def sample_us_estimate_of_prc_compute_stock(prc_compute_stock):
    # Calculate pdf of absolute relative error
    k = -np.log(0.5) / InitialPRCComputeStockParameters.us_intelligence_median_error_in_estimate_of_prc_compute_stock

    u = np.random.uniform(0, 1)

    # Invert the CDF: P(|error|/actual <= x) = 1 - e^(-kx)
    # u = 1 - e^(-kx)
    # e^(-kx) = 1 - u
    # -kx = ln(1 - u)
    # x = -ln(1 - u) / k

    relative_error = -np.log(1 - u) / k

    # Randomly choose direction of error (overestimate or underestimate)
    error_sign = 1 if random.random() > 0.5 else -1
    relative_error = error_sign * relative_error

    # Apply error to actual count
    us_estimate = prc_compute_stock * (1 + relative_error)

    # Ensure estimate is non-negative
    return max(0, us_estimate)


def lr_from_prc_compute_accounting(reported_prc_compute_stock, optimal_diversion_proportion, us_estimate_of_prc_compute_stock):
    """Calculate likelihood ratio from global compute accounting"""
    # Consider two cases:
    # Case 1: Covert project exists of optimal size
    # Case 2: Covert project does not exist

    # Case 1: Covert project exists of optimal size
    # What is the probability that the us_estimate_of_prc_scanners would be what it is in this case?

    true_compute_stock_if_covert_project_exists = reported_prc_compute_stock / (1 - optimal_diversion_proportion)

    # Handle edge cases where the compute stock is zero or near zero
    if true_compute_stock_if_covert_project_exists < 1e-10 or reported_prc_compute_stock < 1e-10:
        return 1.0  # Neutral evidence if no scanners

    us_estimate_absolute_error_if_project_exists = abs(us_estimate_of_prc_compute_stock - true_compute_stock_if_covert_project_exists) / true_compute_stock_if_covert_project_exists

    # PDF of absolute error
    k = -np.log(0.5) / InitialPRCComputeStockParameters.us_intelligence_median_error_in_estimate_of_prc_compute_stock
    p_observe_us_estimate_error_if_project_exists = k * np.exp(-k * us_estimate_absolute_error_if_project_exists)

    # Case 2: Fab does not exist
    true_prc_compute_stock_if_no_project_exists = reported_prc_compute_stock
    us_estimate_absolute_error_if_no_project_exists = abs(us_estimate_of_prc_compute_stock - true_prc_compute_stock_if_no_project_exists) / true_prc_compute_stock_if_no_project_exists

    p_observe_us_estimate_error_if_no_project_exists = k * np.exp(-k * us_estimate_absolute_error_if_no_project_exists)

    if p_observe_us_estimate_error_if_no_project_exists > 0:
        lr = p_observe_us_estimate_error_if_project_exists / p_observe_us_estimate_error_if_no_project_exists
    else:
        lr = 1e6  # Very large if no-fab scenario makes estimate very unlikely

    return lr

def sample_global_compute(year):
    """Sample global compute stock based on year"""
    from util import sample_from_log_normal
    params = InitialPRCComputeStockParameters
    years_since_2025 = year - 2025
    median_total_stock = params.total_global_compute_in_2025 * (params.annual_growth_rate_of_global_compute ** years_since_2025)
    relative_sigma = params.relative_sigma_of_global_compute

    global_compute = sample_from_log_normal(median_total_stock, relative_sigma)

    return global_compute

def sample_reported_global_compute(prc_compute_stock_diverted, global_compute):
    def _sample_unreported_compute_owned_by_non_prc_actors():
        """Sample unreported compute owned by non-PRC actors based on year"""
        from util import sample_from_log_normal
        params = InitialPRCComputeStockParameters
        median = params.median_unreported_compute_owned_by_non_prc_actors = 1e6
        relative_sigma = params.relative_sigma_unreported_compute_owned_by_non_prc_actors = 0.5

        unreported_compute_owned_by_non_prc_actors = sample_from_log_normal(median, relative_sigma)

        return unreported_compute_owned_by_non_prc_actors
    
    unreported_compute_owned_by_non_prc_actors = _sample_unreported_compute_owned_by_non_prc_actors()

    reported_global_compute = global_compute - unreported_compute_owned_by_non_prc_actors - prc_compute_stock_diverted

    print(f"DEBUG: global_compute unreported by non PRC actors: {unreported_compute_owned_by_non_prc_actors}", flush=True)
    print(f"DEBUG: global_compute unreported by PRC: {prc_compute_stock_diverted}", flush=True)
    return reported_global_compute

def get_reported_global_compute_production(year, prc_compute_stock_diverted):
    total_global_compute = InitialPRCComputeStockParameters.total_global_compute_in_2025 * (InitialPRCComputeStockParameters.annual_growth_rate_of_global_compute ** (year - 2025))
    return total_global_compute - prc_compute_stock_diverted

def lr_from_global_compute_production_accounting(reported_historical_global_compute_production, reported_global_compute, reported_prc_compute_stock, optimal_diversion_proportion):

    discrepency = reported_global_compute - reported_historical_global_compute_production

    # Case 1: There's no covert PRC project
    params = InitialPRCComputeStockParameters
    relative_sigma = params.relative_sigma_unreported_compute_owned_by_non_prc_actors
    sigma_log = np.sqrt(np.log(1 + relative_sigma**2))
    mu_log = np.log(params.median_unreported_compute_owned_by_non_prc_actors)

    # Lognormal PDF
    if discrepency > 0:
        p_discrepency_if_no_prc_project = (1 / (discrepency * sigma_log * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(discrepency) - mu_log) / sigma_log)**2)
    else:
        p_discrepency_if_no_prc_project = 1e-10  # Very small probability for non-positive values
    

    # Case 2: There is a covert PRC project
    estimated_prc_compute_stock = reported_prc_compute_stock / (1 - optimal_diversion_proportion)
    estimated_computes_diverted_to_prc_project = estimated_prc_compute_stock - reported_prc_compute_stock
    estimated_compute_unreported_by_non_prc_actors = discrepency - estimated_computes_diverted_to_prc_project

    # Lognormal PDF
    if estimated_compute_unreported_by_non_prc_actors > 0:
        p_discrepency_if_prc_project = (1 / (estimated_compute_unreported_by_non_prc_actors * sigma_log * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(estimated_compute_unreported_by_non_prc_actors) - mu_log) / sigma_log)**2)
    else:
        p_discrepency_if_prc_project = 1e-10  # Very small probability for non-positive values

    lr = p_discrepency_if_prc_project / p_discrepency_if_no_prc_project

    return lr

def sample_hazard_rate_multiplier() -> float:
    """Sample a multiplier for hazard rates from a metalog distribution.

    Returns a multiplier that will be applied to both initial_hazard_rate and
    increase_of_hazard_rate_per_year to create correlated uncertainty.
    """
    from util import sample_from_metalog_3term_semi_bounded
    params = SurvivalRateParameters

    # Compute absolute percentiles from the ratios
    p25 = params.hazard_rate_p25_relative_to_p50
    p50 = 1.0  # Median multiplier is 1.0
    p75 = params.hazard_rate_p75_relative_to_p50

    # Sample multiplier from semi-bounded metalog distribution (lower bound 0, no upper bound)
    return sample_from_metalog_3term_semi_bounded(p25, p50, p75)

def sample_hazard_rates() -> tuple[float, float]:
    """Sample both initial hazard rate and increase rate using a common multiplier.

    Returns:
        tuple: (initial_hazard_rate, increase_of_hazard_rate_per_year)
    """
    params = SurvivalRateParameters
    multiplier = sample_hazard_rate_multiplier()

    initial_hazard_rate = params.initial_hazard_rate_p50 * multiplier
    increase_of_hazard_rate_per_year = params.increase_of_hazard_rate_per_year_p50 * multiplier

    return initial_hazard_rate, increase_of_hazard_rate_per_year


class Chip():
    def __init__(self, h100e_per_chip: float, energy_consumed_per_chip_GW: float, intra_chip_memory_bandwidth_tbps: float = 8, inter_chip_memory_bandwidth_tbps: float = 1.8):
        self.h100e_per_chip = h100e_per_chip
        self.energy_consumed_per_chip_GW = energy_consumed_per_chip_GW
        self.intra_chip_memory_bandwidth_tbps = intra_chip_memory_bandwidth_tbps
        self.inter_chip_memory_bandwidth_tbps = inter_chip_memory_bandwidth_tbps
    
class Compute():
    def __init__(self, chip_counts: Dict[Chip, float]):
        self.chip_counts = chip_counts  # Dict of Chip to number of chips
    
    def total_h100e(self) -> float:
        return sum(chip.h100e_per_chip * count for chip, count in self.chip_counts.items())
    
    def total_energy_requirements_GW(self) -> float:
        return sum(chip.energy_consumed_per_chip_GW * count for chip, count in self.chip_counts.items())

class PRCDarkComputeStock():

    def __init__(self, agreement_year, proportion_of_initial_compute_stock_to_divert, optimal_proportion_of_initial_compute_stock_to_divert):
        self.agreement_year = agreement_year
        self.initial_prc_stock = sample_initial_prc_compute_stock(agreement_year)
        self.initial_prc_dark_compute = self.initial_prc_stock * proportion_of_initial_compute_stock_to_divert
        self.dark_compute_added_per_year = {}
        self.dark_compute_added_per_year[agreement_year] = self.initial_prc_dark_compute 
        self.us_estimate_of_prc_stock = sample_us_estimate_of_prc_compute_stock(self.initial_prc_stock)
        self.lr_from_prc_compute_accounting = lr_from_prc_compute_accounting(
            reported_prc_compute_stock=self.initial_prc_stock - self.initial_prc_dark_compute,
            optimal_diversion_proportion=optimal_proportion_of_initial_compute_stock_to_divert,
            us_estimate_of_prc_compute_stock=self.us_estimate_of_prc_stock
        )
        self.global_compute = sample_global_compute(agreement_year)
        print(f"DEBUG: global compute: {self.global_compute}", flush=True)

        self.reported_global_compute = sample_reported_global_compute(self.initial_prc_dark_compute, self.global_compute)
        print(f"DEBUG: reported global compute: {self.reported_global_compute}", flush=True)

        self.reported_historical_global_compute_production = get_reported_global_compute_production(agreement_year, self.initial_prc_dark_compute)
        self.lr_from_global_compute_production_accounting = lr_from_global_compute_production_accounting(
            reported_historical_global_compute_production= self.reported_historical_global_compute_production,
            reported_global_compute=self.reported_global_compute,
            reported_prc_compute_stock=self.initial_prc_stock - self.initial_prc_dark_compute,
            optimal_diversion_proportion=optimal_proportion_of_initial_compute_stock_to_divert
        )
        # Sample both hazard rates using the correlated multiplier
        self.initial_hazard_rate, self.increase_in_hazard_rate_per_year = sample_hazard_rates()
    
    def add_dark_compute(self, year : float, additional_dark_compute : float):
        self.dark_compute_added_per_year[year] = additional_dark_compute
    
    def operational_and_nonoperational_dark_compute(self, year : float):
        total = sum(self.dark_compute_added_per_year[y] for y in self.dark_compute_added_per_year if y <= year)
        print(f"DEBUG operational_and_nonoperational_dark_compute(year={year}): total={total}, dark_compute_added_per_year={dict(list(self.dark_compute_added_per_year.items())[:5])}", flush=True)
        return total

    def annual_hazard_rate_after_years_of_life(self, years_of_life: float) -> float:
        return self.initial_hazard_rate + self.increase_in_hazard_rate_per_year * years_of_life

    def operational_dark_compute(self, year : float):
        total_surviving_compute = 0
        debug_info = []
        for y in self.dark_compute_added_per_year:
            years_of_life = year - y
            # Skip if compute hasn't been added yet (negative years of life)
            if years_of_life < 0:
                continue
            # Cumulative hazard: integral of hazard rate from 0 to years_of_life
            # hazard_rate(t) = initial_hazard_rate + increase_in_hazard_rate_per_year * t
            # cumulative_hazard = integral_0^T (initial + increase * t) dt = initial*T + increase*T^2/2
            cumulative_hazard = self.initial_hazard_rate * years_of_life + self.increase_in_hazard_rate_per_year * years_of_life**2 / 2
            survival_rate = np.exp(-cumulative_hazard)
            surviving_compute_from_year_y = self.dark_compute_added_per_year[y] * survival_rate
            total_surviving_compute += surviving_compute_from_year_y
            if len(debug_info) < 3:  # Only log first 3 entries
                debug_info.append(f"y={y}, years_of_life={years_of_life:.2f}, cumulative_hazard={cumulative_hazard:.4f}, survival_rate={survival_rate:.4f}, surviving_compute={surviving_compute_from_year_y:.2f}, compute_added={self.dark_compute_added_per_year[y]:.2f}")

        if debug_info:
            print(f"DEBUG operational_dark_compute(year={year}): total_surviving={total_surviving_compute:.4f}, initial_hazard={self.initial_hazard_rate}, increase_per_year={self.increase_in_hazard_rate_per_year:.4f}", flush=True)
            for info in debug_info:
                print(f"  {info}", flush=True)
        return total_surviving_compute