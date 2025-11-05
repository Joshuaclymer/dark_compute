from dataclasses import dataclass
from full_model import ProcessNode
import numpy as np
from scipy.interpolate import interp1d
from scipy import stats
from typing import List
import random

class FabModelParameters:
    proportion_of_diverted_sme_with_50p_chance_of_detection = 0.14

    # Detection via other strategies - anchor points (from detection_probability_vs_num_workers/model_bayesian.py)
    # These define the relationship between number of workers and mean detection time
    # Format: (number_of_workers, mean_detection_time_in_years)
    mean_detection_time_for_100_workers = 6.95   # Small operation: ~7 years to detect
    mean_detection_time_for_1000_workers = 3.42  # Medium operation: ~3.4 years to detect

    # Variance parameter: describes uncertainty in detection time given number of workers
    # Higher variance means more uncertainty about when detection will occur
    # This represents the inherent randomness in intelligence collection effectiveness
    variance_of_detection_time_given_num_workers = 3.880

    # Fab production capacity parameters (from fab_production/)
    # Production depends on both operating labor and photolithography scanners (SME)
    # Labor relationship: wafers_per_month = wafers_per_month_per_worker * operating_labor
    # Linear fit through origin from historical fab data (fab_production/labor_vs_fab_production.py)
    wafers_per_month_per_worker = 24.64  # Each worker produces ~25 wafers per month

    # SME relationship: each photolithography scanner can process ~1000 wafers per month
    # Based on ASML DUV specs: 250 wafers/hour, 80 patterning steps, 50% utilization
    # Calculation: 0.5 * 180,000 wafers_per_month / 80 steps = 1000 wafers/month
    # Source: fab_production/production_vs_sme.py
    wafers_per_month_per_lithography_scanner = 1000

    # Fab construction time parameters (from fab_construction_time/)
    # Anchor points defining relationship between capacity and construction time for covert fabs
    # Based on historical data from fab_construction_time/construction_time_plot.py
    construction_time_for_5k_wafers_per_month = 1.40    # Small fab: ~1.4 years
    construction_time_for_100k_wafers_per_month = 2.41  # Large fab: ~2.4 years

    # Construction labor requirements (from fab_construction_time/)
    # Combined parameter from cost_of_fab_plot.py and construction_time_vs_labor.py:
    # - Cost: $0.000141B per wafer/month capacity
    # - Labor: 100 workers per $1B construction cost
    # Combined: 100 * 0.000141 * 1000 = 14.1 workers per 1000 wafers/month
    construction_workers_per_1000_wafers_per_month = 14.1

    # Chip production parameters
    # Number of H100-sized chips that can be produced per wafer
    h100_sized_chips_per_wafer = 28

    # Transistor density scaling (Moore's Law)
    # When process node halves, transistor density increases by 2^transistor_density_scaling_exponent
    # With exponent = 1.49, halving node size increases density by 2^1.49 ≈ 2.81×
    transistor_density_scaling_exponent = 1.49
    h100_process_node_nm = 4  # H100 was fabricated on 4nm node

    # Chip architecture efficiency improvement over time
    # Architectures improve at a pace of 1.23× per year
    # H100 serves as the reference point (released in 2022)
    architecture_efficiency_improvement_per_year = 1.23
    h100_release_year = 2022

    # PRC lithography scanner production ramp-up (from china_domestic_fab_capability/lithography_sales_plot.py)
    # Linear trend: production_per_year = additional_per_year * years_since_localization + first_year_production
    # Based on estimated PRC ramp-up: 20 scanners at year 0, 100 scanners at year 5
    prc_additional_lithography_scanners_produced_per_year = 16.0  # Additional scanners per year
    prc_lithography_scanners_produced_in_first_year = 20.0  # Production in year of localization

    # Self-sufficiency probability data (from china_domestic_fab_capability/forecasts.py)
    # Represents probability of >90% localization for each node
    # Data structure: {process_node: [(year, p_selfsufficiency), ...]}
    Probability_of_90p_PRC_localization_at_node = {
        ProcessNode.nm130: [(2026, 0.80), (2028, 0.80), (2031, 0.80)],
        ProcessNode.nm28: [(2026, 0.0), (2028, 0.07), (2031, 0.25)],
        ProcessNode.nm14: [(2026, 0.0), (2028, 0.03), (2031, 0.10)],
        ProcessNode.nm7: [(2026, 0.0), (2028, 0.01), (2031, 0.06)]
    }

# ============================================================================
# Helper functions for estimating fab compute production
# ============================================================================

def estimate_wafer_starts_per_month(
    operation_labor: float,
    number_of_prc_lithography_scanners_devoted_to_fab: float
) -> float:
    """
    Estimates wafer starts per month using fixed-proportions production function.

    Production is limited by the more constraining factor between labor and equipment.
    This represents the reality that you need both workers and scanners to produce wafers.

    Args:
        operation_labor: Number of workers dedicated to fab operation
        number_of_prc_lithography_scanners_devoted_to_fab: Number of lithography scanners allocated to the fab

    Returns:
        float: Estimated wafer starts per month
    """
    wafers_per_month_achievable_given_operation_labor = FabModelParameters.wafers_per_month_per_worker * operation_labor
    wafers_per_month_achievable_given_lithography_scanners = FabModelParameters.wafers_per_month_per_lithography_scanner * number_of_prc_lithography_scanners_devoted_to_fab

    return min(wafers_per_month_achievable_given_operation_labor, wafers_per_month_achievable_given_lithography_scanners)

def estimate_construction_duration(
    wafer_starts_per_month: float,
    construction_labor: float
) -> float:
    """
    Estimates fab construction duration using fixed-proportions production function.

    Construction time is determined by the more constraining factor between:
    1. Fab capacity (larger fabs take longer to build)
    2. Construction labor (fewer workers means longer construction)

    Args:
        wafer_starts_per_month: Production capacity of the fab in wafers per month
        construction_labor: Number of workers dedicated to construction

    Returns:
        float: Construction duration in years
    """
    def _construction_duration_given_wafer_capacity(wafer_starts_per_month: float) -> float:
        """
        Estimates construction duration based solely on fab capacity.

        Derives parameters from anchor points using the model:
        construction_duration = slope * log10(capacity) + intercept
        """
        # Given two anchor points (capacity1, time1) and (capacity2, time2), we solve:
        #   time1 = slope * log10(capacity1) + intercept
        #   time2 = slope * log10(capacity2) + intercept
        # Taking the difference: time2 - time1 = slope * (log10(capacity2) - log10(capacity1))
        # Therefore: slope = (time2 - time1) / (log10(capacity2) - log10(capacity1))
        # And then: intercept = time1 - slope * log10(capacity1)

        capacity1 = 5000
        time1 = FabModelParameters.construction_time_for_5k_wafers_per_month
        capacity2 = 100000
        time2 = FabModelParameters.construction_time_for_100k_wafers_per_month

        slope = (time2 - time1) / (np.log10(capacity2) - np.log10(capacity1))
        intercept = time1 - slope * np.log10(capacity1)

        return slope * np.log10(wafer_starts_per_month) + intercept

    # Step 1: Calculate construction duration given wafer capacity
    construction_duration = _construction_duration_given_wafer_capacity(wafer_starts_per_month)

    # Step 2: Calculate construction labor requirement given wafer capacity
    # Heuristic: 14.1 workers per 1000 wafers/month of capacity needed for baseline construction time
    construction_labor_requirement_given_wafer_capacity = (
        FabModelParameters.construction_workers_per_1000_wafers_per_month / 1000
    ) * wafer_starts_per_month

    # Step 3: Check if construction labor is the constraining factor
    is_construction_labor_constraining = construction_labor < construction_labor_requirement_given_wafer_capacity

    # Step 4: If labor is constraining, extend construction duration proportionally
    if is_construction_labor_constraining:
        construction_duration *= (construction_labor_requirement_given_wafer_capacity / construction_labor)

    return construction_duration

def estimate_transistor_density_relative_to_h100(process_node_nm: float) -> float:
    """
    Estimates transistor density relative to H100 based on process node.

    Uses Moore's Law scaling: when the process node halves, transistor density increases
    by 2^(scaling_exponent). The H100 was fabricated on the 4nm node, which serves as
    the reference point.

    Scaling formula:
        density_ratio = (h100_node / fab_node)^scaling_exponent

    Args:
        process_node_nm: Process node of the fab in nanometers (e.g., 7, 14, 28)

    Returns:
        float: Transistor density relative to H100 (1.0 = same as H100, >1 = denser, <1 = less dense)
    """
    # Estimate density ratio using Moore's Law scaling
    density_relative_to_h100 = (
        FabModelParameters.h100_process_node_nm / process_node_nm
    ) ** FabModelParameters.transistor_density_scaling_exponent

    return density_relative_to_h100

def estimate_architecture_efficiency_relative_to_h100(year: float) -> float:
    """
    Estimates chip architecture efficiency relative to H100 based on year.

    Architecture improvements compound at 1.23× per year. The H100 (released in 2022)
    serves as the reference point with efficiency = 1.0.

    Args:
        year: The year to calculate architecture efficiency for

    Returns:
        float: Architecture efficiency relative to H100 (1.0 = H100 in 2022, >1 = better, <1 = worse)
    """
    years_since_h100 = year - FabModelParameters.h100_release_year
    efficiency_relative_to_h100 = FabModelParameters.architecture_efficiency_improvement_per_year ** years_since_h100

    return efficiency_relative_to_h100

def sample_year_prc_achieved_node_localization(process_node: ProcessNode) -> float:
    """
    Randomly samples the year when PRC first achieved >90% localization for a given process node.

    Uses the probability curves in FabModelParameters.Probability_of_90p_PRC_localization_at_node
    as a CDF to randomly sample when localization was achieved. This treats the probability
    as the cumulative probability that localization has been achieved by that year.

    Args:
        process_node: The process node (e.g., ProcessNode.nm7, ProcessNode.nm28)

    Returns:
        float: Randomly sampled year when PRC achieved localization
    """
    if process_node not in FabModelParameters.Probability_of_90p_PRC_localization_at_node:
        raise ValueError(f"Process node {process_node} not found in Probability_of_90p_PRC_localization_at_node parameters")

    # Define the CDF for probability that PRC achieves localization over time
    data = FabModelParameters.Probability_of_90p_PRC_localization_at_node[process_node]
    data_years = np.array([point[0] for point in data])
    data_probabilities = np.array([point[1] for point in data])

    # Fit a quadratic curve: p = a * year^2 + b * year + c
    # Then numerically invert to get year as a function of probability
    coeffs = np.polyfit(data_years, data_probabilities, deg=2)
    prob_curve = np.poly1d(coeffs)

    # Generate dense year range and evaluate probabilities
    # Extend beyond data range to ensure we cover p=1.0
    year_min = data_years.min()
    year_max = data_years.max() + 50  # Extend 50 years beyond last data point
    dense_years = np.linspace(year_min, year_max, 1000)
    dense_probabilities = prob_curve(dense_years)

    # Clip probabilities to [0, 1] range
    dense_probabilities = np.clip(dense_probabilities, 0, 1)

    # Create CDF by finding years for 100 evenly spaced probabilities
    cdf_probabilities = np.linspace(0, 1, 100)
    cdf_years = np.interp(cdf_probabilities, dense_probabilities, dense_years)

    def _sample_from_cdf(years: np.ndarray, probabilities: np.ndarray) -> float:
        """
        Samples a year from a CDF by sampling a random probability and finding the corresponding year.

        Args:
            years: Array of years (x-axis of CDF)
            probabilities: Array of cumulative probabilities (y-axis of CDF)

        Returns:
            float: Sampled year
        """
        random_cdf_value = random.random()

        # Find the closest probability in the CDF and return the corresponding year
        idx = np.searchsorted(probabilities, random_cdf_value)

        # Clamp to valid range (should always be valid since CDF goes from 0 to 1)
        idx = min(idx, len(years) - 1)
        return float(years[idx])

    # Sample from the CDF
    return _sample_from_cdf(cdf_years, cdf_probabilities)

def estimate_total_prc_lithography_scanners_for_node(current_year: float, localization_year: float) -> float:
    """
    Estimates total accumulated PRC lithography scanners for a given node at a given year.

    Uses the PRC production ramp-up trend from china_domestic_fab_capability/lithography_sales_plot.py.
    Annual production follows: production_per_year = slope * years_since_localization + intercept

    This function accumulates the yearly production to get total scanners:
    Total = sum over all years of (intercept + slope * year_index)

    Args:
        current_year: The year to estimate total scanners for
        localization_year: The year when PRC achieved 90% localization for this node

    Returns:
        float: Total accumulated number of lithography scanners PRC has for this node
    """
    # Estimate years since achieving localization
    years_since_localization = current_year - localization_year

    # If haven't achieved localization yet, return 0
    if years_since_localization < 0:
        return 0.0

    # Accumulate scanners using analytical formula for linear ramp-up
    # Total = sum from i=0 to n of (intercept + i*slope)
    #       = intercept * (n+1) + slope * n*(n+1)/2
    n = years_since_localization
    total_scanners = (FabModelParameters.prc_lithography_scanners_produced_in_first_year * (n + 1) +
                     FabModelParameters.prc_additional_lithography_scanners_produced_per_year * n * (n + 1) / 2)

    return total_scanners

# ============================================================================
# Helper functions for estimating detection probability
# ============================================================================

def p_detection_via_inventory_accounting(proportion_of_prc_sme_diverted_to_fab: float) -> float:
    """
    Probability of detection via inventory accounting - tracking missing SME (photolithography scanners).

    Based on historical intelligence error data showing relationship between proportion of
    diverted equipment and detection probability. Uses bounded exponential model derived from
    intelligence_error_bars/diversion_vs_detection.py.

    Args:
        proportion_of_prc_sme_diverted_to_fab: Proportion (0-1) of PRC's domestic lithography
            scanners allocated to this covert fab

    Returns:
        float: Probability (0-1) that intelligence agencies detect the fab via inventory accounting
    """
    # The function f(x) = 1 - e^(-kx) maps proportion of diverted SME to detection probability
    # The rate constant k is calibrated so that 50% detection occurs at the specified proportion
    k = -np.log(0.5) / FabModelParameters.proportion_of_diverted_sme_with_50p_chance_of_detection
    return 1 - np.exp(-k * proportion_of_prc_sme_diverted_to_fab)

def p_detection_via_procurement_accounting(
    construction_start_year: float,
    year_process_node_achieved_90p_prc_localization: float
) -> float:
    """
    Probability of detection via procurement accounting - tracking purchases of foreign SME.

    If the fab's process node has achieved >90% domestic localization by construction start,
    then most equipment can be sourced domestically without triggering foreign procurement alerts.
    Otherwise, the need to import foreign SME makes detection highly likely.

    Args:
        construction_start_year: The year when fab construction starts
        year_process_node_achieved_90p_prc_localization: The year when PRC achieved 90% localization for this node

    Returns:
        float: Probability (0 or 1) that intelligence agencies detect the fab via procurement tracking
    """
    # Binary outcome: if localization achieved before construction start (p=0), otherwise (p=1)
    return float(year_process_node_achieved_90p_prc_localization > construction_start_year)

def p_detection_via_other_strategies(
    total_labor: int,
    years_since_construction_start: float
) -> float:
    """
    Probability of detection via other intelligence strategies (HUMINT, SIGINT, etc.).

    Based on historical nuclear program detection data, models how detection probability
    increases with both the number of workers involved and time elapsed since construction start.
    More workers create more HUMINT opportunities and communications to intercept.

    Model from detection_probability_vs_num_workers/model_bayesian.py, calibrated on
    historical covert nuclear facility detection cases.

    Args:
        total_labor: Total number of workers involved (construction + operation)
        years_since_construction_start: Years elapsed since construction began

    Returns:
        float: Probability (0-1) that intelligence agencies detect the fab via other strategies
    """
    if total_labor > 0 and years_since_construction_start > 0:
        # Derive model parameters from anchor points for the detection model: μ(x) = A / log10(x)^B
        # Given two anchor points (x1, μ1) and (x2, μ2), we solve:
        #   μ1 = A / log10(x1)^B
        #   μ2 = A / log10(x2)^B
        # Taking the ratio: μ1/μ2 = (log10(x2) / log10(x1))^B
        # Therefore: B = log(μ1/μ2) / log(log10(x2)/log10(x1))
        # And then: A = μ1 * log10(x1)^B

        x1 = 100
        mu1 = FabModelParameters.mean_detection_time_for_100_workers
        x2 = 1000
        mu2 = FabModelParameters.mean_detection_time_for_1000_workers

        B = np.log(mu1 / mu2) / np.log(np.log10(x2) / np.log10(x1))
        A = mu1 * (np.log10(x1) ** B)

        # Mean detection time: μ(x) = A / log10(x)^B
        mu = A / (np.log10(total_labor) ** B)

        # Gamma distribution parameters: mean = k*θ, variance = k*θ²
        # Given: mean = μ, variance = σ²*μ (proportional to mean)
        # Solving: θ = σ², k = μ/σ²
        theta = FabModelParameters.variance_of_detection_time_given_num_workers
        k = mu / FabModelParameters.variance_of_detection_time_given_num_workers

        # P(detection by time t) = CDF of Gamma distribution at t
        return stats.gamma.cdf(years_since_construction_start, a=k, scale=theta)
    else:
        return 0.0

# ============================================================================
# PRC covert fab core logic
# ============================================================================
@dataclass
class PRCCovertFab:
    # Properties that are used to determine compute production
    construction_start_year : float
    construction_duration : float
    wafer_starts_per_month : float
    h100_sized_chips_per_wafer : float
    transistor_density_relative_to_h100 : float

    # Properties that are used to determine detection probability
    process_node : ProcessNode
    year_process_node_achieved_90p_prc_localization : float
    proportion_of_prc_lithography_scanners_devoted_to_fab : float
    construction_labor : int
    operation_labor : int

    def __init__(
            self,
            construction_start_year : float,
            construction_labor : float,
            process_node : float,
            proportion_of_prc_lithography_scanners_devoted_to_fab : float,
            operation_labor : float,
    ):
        # Sample the year when this process node achieved 90% PRC localization
        # This is used consistently for both detection probability and machine count calculations
        self.year_process_node_achieved_90p_prc_localization = sample_year_prc_achieved_node_localization(process_node)

        # Estimate total PRC lithography scanners for this node at construction start
        self.total_prc_lithography_scanners_for_node = estimate_total_prc_lithography_scanners_for_node(
            current_year=construction_start_year,
            localization_year=self.year_process_node_achieved_90p_prc_localization
        )

        # Properties that are used to determine compute production
        self.construction_start_year = construction_start_year

        # Estimate number of scanners devoted to fab from proportion
        number_of_prc_lithography_scanners_devoted_to_fab = proportion_of_prc_lithography_scanners_devoted_to_fab * self.total_prc_lithography_scanners_for_node

        # Estimate wafer starts per month using fixed-proportions production function
        self.wafer_starts_per_month = estimate_wafer_starts_per_month(
            operation_labor=operation_labor,
            number_of_prc_lithography_scanners_devoted_to_fab=number_of_prc_lithography_scanners_devoted_to_fab
        )

        # Estimate construction duration based on capacity and construction labor
        self.construction_duration = estimate_construction_duration(
            wafer_starts_per_month=self.wafer_starts_per_month,
            construction_labor=construction_labor
        )

        self.h100_sized_chips_per_wafer = FabModelParameters.h100_sized_chips_per_wafer
        self.transistor_density_relative_to_h100 = estimate_transistor_density_relative_to_h100(process_node)

        # Properties that are used to determine detection probability
        self.process_node = process_node
        self.proportion_of_prc_lithography_scanners_devoted_to_fab = proportion_of_prc_lithography_scanners_devoted_to_fab
        self.construction_labor = construction_labor
        self.operation_labor = operation_labor

    def is_operational(self, year):
        if self.construction_start_year != None: # Construction start year can be none if the PRC never builds a covert fab
            return year > self.construction_start_year + self.construction_duration
        else:
            return False

    def h100e_produced_per_month(self, year):
        chip_architecture_efficiency_relative_to_h100 = estimate_architecture_efficiency_relative_to_h100(year)
        return self.is_operational(self, year) \
            * self.wafer_starts_per_month * self.h100_sized_chips_per_wafer * self.transistor_density_relative_to_h100 * chip_architecture_efficiency_relative_to_h100

    def US_intelligence_estimate_of_probability_fab_exists(self, year):
        """
        Overall probability that US intelligence estimates the fab exists.

        Combines three independent detection methods:
        1. Inventory accounting (missing domestic SME)
        2. Procurement accounting (foreign SME purchases)
        3. Other strategies (HUMINT, SIGINT based on workers and time)

        Uses probability theory for independent events: P(A or B or C) = 1 - P(not A) * P(not B) * P(not C)

        Args:
            year: Current year to evaluate detection probability

        Returns:
            float: Overall probability (0-1) that intelligence agencies believe the fab exists
        """
        # Inventory accounting: Determine whether Chinese SME has gone missing
        # Tracks whether domestic PRC lithography scanners are unaccounted for
        p_inventory = p_detection_via_inventory_accounting(
            proportion_of_prc_sme_diverted_to_fab=self.proportion_of_prc_lithography_scanners_devoted_to_fab
        )

        # Procurement accounting: Determine whether the SME China has imported can be accounted for
        # If PRC hasn't achieved domestic localization, foreign scanner imports would be detectable
        p_procurement = p_detection_via_procurement_accounting(
            construction_start_year=self.construction_start_year,
            year_process_node_achieved_90p_prc_localization=self.year_process_node_achieved_90p_prc_localization
        )

        # Other: Leverage signals intelligence, human intelligence, and satellite imagery to detect a chip fab operation
        # Detection probability increases with number of workers and time elapsed
        p_other = p_detection_via_other_strategies(
            total_labor=self.construction_labor + self.operation_labor,
            years_since_construction_start=year - self.construction_start_year
        )

        return 1 - (1 - p_inventory) * (1 - p_procurement) * (1 - p_other)
