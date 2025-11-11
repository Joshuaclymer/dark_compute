import numpy as np
import random
from typing import List
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

    # Median survival rate
    initial_hazard_rate = 0.02
    increase_of_hazard_rate_per_year = 0.0416
    increase_of_hazard_rate_per_year_relative_sigma = 0.2


def sample_initial_prc_compute_stock(year):
    """Sample initial PRC compute stock based on year and proportion diverted to covert project"""
    params = InitialPRCComputeStockParameters
    years_since_2025 = year - 2025
    total_stock_mu = params.total_prc_compute_stock_in_2025 * (params.annual_growth_rate_of_prc_compute_stock ** years_since_2025)
    relative_sigma = params.relative_sigma_of_prc_compute_stock

    sigma_log = np.sqrt(np.log(1 + relative_sigma**2))
    mu_log = np.log(total_stock_mu) - 0.5 * sigma_log**2

    total_stock_sample = np.random.lognormal(mu_log, sigma_log)

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
    params = InitialPRCComputeStockParameters
    years_since_2025 = year - 2025
    total_stock_mu = params.total_global_compute_in_2025 * (params.annual_growth_rate_of_global_compute ** years_since_2025)
    relative_sigma = params.relative_sigma_of_global_compute

    sigma_log = np.sqrt(np.log(1 + relative_sigma**2))
    mu_log = np.log(total_stock_mu) - 0.5 * sigma_log**2

    global_compute = np.random.lognormal(mu_log, sigma_log)

    return global_compute

def sample_reported_global_compute(prc_compute_stock_diverted, global_compute):
    def _sample_unreported_compute_owned_by_non_prc_actors():
        """Sample unreported compute owned by non-PRC actors based on year"""
        params = InitialPRCComputeStockParameters
        mu = params.median_unreported_compute_owned_by_non_prc_actors = 1e6
        relative_sigma = params.relative_sigma_unreported_compute_owned_by_non_prc_actors = 0.5

        sigma_log = np.sqrt(np.log(1 + relative_sigma**2))
        mu_log = np.log(mu) - 0.5 * sigma_log**2

        unreported_compute_owned_by_non_prc_actors = np.random.lognormal(mu_log, sigma_log)

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
    mu_log = np.log(params.median_unreported_compute_owned_by_non_prc_actors) - 0.5 * sigma_log**2

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

def sample_increase_in_hazard_rate_per_year() -> float:
    params = SurvivalRateParameters
    increase_of_hazard_rate_per_year = params.increase_of_hazard_rate_per_year
    relative_sigma = params.increase_of_hazard_rate_per_year_relative_sigma
    
    # Calculate log-normal parameters for the increase rate
    # For a log-normal distribution with mean μ and relative_sigma σ_rel:
    # sigma_log = sqrt(log(1 + relative_sigma²))
    # mu_log = log(mean) - sigma_log²/2
    sigma_log = np.sqrt(np.log(1 + relative_sigma**2))
    mu_log = np.log(increase_of_hazard_rate_per_year) - 0.5 * sigma_log**2
    
    # Sample increase rates from log-normal distribution
    sampled_increase_of_rates_per_year = np.random.lognormal(mu_log, sigma_log, size=1)
    
    return sampled_increase_of_rates_per_year[0]


class Stock(ABC):
    dark_compute_over_time : dict[float, float]  # year -> cumulative H100e of dark compute
    dark_compute : float  # Total current dark compute stock

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
        self.increase_in_hazard_rate_per_year = sample_increase_in_hazard_rate_per_year()
    
    def add_dark_compute(self, year : float, additional_dark_compute : float):
        self.dark_compute_added_per_year[year] = additional_dark_compute
    
    def operational_and_nonoperational_dark_compute(self, year : float):
        return sum(self.dark_compute_added_per_year[y] for y in self.dark_compute_added_per_year if y <= year)

    def annual_hazard_rate_after_years_of_life(self, years_of_life: float) -> float:
        return SurvivalRateParameters.initial_hazard_rate + self.increase_in_hazard_rate_per_year * years_of_life

    def operational_dark_compute(self, year : float):
        total_surviving_compute = 0
        for y in self.dark_compute_added_per_year:
            years_of_life = year - y
            cumulative_hazard = SurvivalRateParameters.initial_hazard_rate * y + self.increase_in_hazard_rate_per_year * years_of_life**2 / 2
            surviving_compute_from_year_y = np.exp(-cumulative_hazard)
            total_surviving_compute += surviving_compute_from_year_y
        return total_surviving_compute