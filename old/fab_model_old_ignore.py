import numpy as np
from scipy import stats
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
import random
from backend_model.util import sample_from_log_normal
from backend.classes.dark_compute_stock import Chip, Compute
from backend_model.util import lr_vs_num_workers
from backend_model.paramaters import ProcessNode, FabModelParameters

"""
PRC Covert Semiconductor Fab Model

This model estimates the compute production capacity and detection probability of a hypothetical
covert semiconductor fabrication facility operated by the People's Republic of China (PRC).

MODEL INPUTS
------------
The model takes five parameters that define a covert fab scenario:

1. construction_start_year (float)
   - Year when fab construction begins (e.g., 2026)

2. construction_labor (float)
   - Number of workers dedicated to building the fab (e.g., 500)

3. process_node (ProcessNode or str)
   - Manufacturing process node in nanometers (e.g., ProcessNode.nm28)
   - Options: nm130, nm28, nm14, nm7, or "best_available_indigenously"
   - If "best_available_indigenously" is specified, the model samples localization years
     for all nodes and selects the most advanced node available at construction start

4. proportion_of_prc_lithography_scanners_devoted_to_fab (float)
   - Fraction of PRC's domestic lithography scanners allocated to this fab (e.g., 0.10 = 10%)

5. operation_labor (float)
   - Number of workers dedicated to operating the fab once built (e.g., 2000)

MODEL OUTPUTS
-------------
Given a specific year, the model provides two key outputs:

1. compute_produced_per_month(year) -> Compute
   - Compute object containing chip specifications and monthly production count
   - Returns empty Compute object if fab not yet operational

2. likelihood_ratio_from_evidence(year) -> float
   - Likelihood ratio from intelligence evidence about the fab at the specified year
   - Multiplies three independent likelihood ratios from different intelligence sources
   - Can be combined with a prior to estimate detection probability

================================================================================
COMPUTE PRODUCTION MODEL
================================================================================

The compute production rate (H100-equivalents per month) is calculated as:

    H100e_per_month = is_operational × wafer_starts_per_month × compute_per_wafer

Where each factor is determined as follows:

Factor 1: Is Operational (binary: 0 or 1)
------------------------------------------
The fab is operational if current year > construction_start_year + construction_duration.

Construction duration depends on two factors via fixed-proportions production function:
  a) Fab capacity: Larger fabs take longer to build
     - Uses logarithmic scaling: duration ∝ log10(wafer_starts_per_month)
     - Calibrated from historical data: 1.4 years for 5k wafers/month, 2.4 years for 100k wafers/month

  b) Construction labor: More workers reduce construction time
     - Baseline requirement: 14.1 workers per 1000 wafers/month capacity
     - If actual labor < baseline, duration extends proportionally
     - Example: Half the required labor doubles construction time

Source: fab_construction_time/

Factor 2: Wafer Starts per Month
---------------------------------
Production capacity is determined by fixed-proportions production function:

    wafer_starts_per_month = min(labor_capacity, equipment_capacity)

Where:
  - labor_capacity = 24.64 × operating_labor
    (Each worker can produce ~25 wafers per month)

  - equipment_capacity = 1000 × number_of_lithography_scanners
    (Each scanner can process ~1000 wafers per month)
    Based on ASML DUV specs: 250 wafers/hour, 80 patterning steps, 50% utilization

The bottleneck input (labor or equipment) constrains overall capacity.

Number of lithography scanners available depends on:
  a) Total PRC domestic scanners for this process node
     - Determined by when PRC achieved 90% localization (sampled from probability curves)
     - Linear production ramp-up after localization: 20 in year 0, +16 per year
     - Accumulated over years since localization

  b) Proportion of PRC scanners devoted to this fab
     - User-specified parameter (e.g., 10% of national capacity)

Source: fab_production/, china_domestic_fab_capability/

Factor 3: Compute per Wafer (H100-equivalents per wafer)
---------------------------------------------------------
Converts physical wafers into H100-equivalent compute:

    compute_per_wafer = chips_per_wafer × transistor_density_ratio × architecture_efficiency

Where:
  a) Chips per wafer = 28
     (Number of H100-sized chips that fit on one wafer)

  b) Transistor density ratio (relative to H100's 4nm node)
     - Moore's Law scaling: density_ratio = (4nm / process_node)^1.49
     - Exponent of 1.49 means halving node size increases density by 2^1.49 ≈ 2.81×
     - Examples: 28nm node → 0.035× H100 density, 7nm node → 0.56× H100 density

  c) Architecture efficiency (relative to H100 released in 2022)
     - Chip architectures improve at 1.23× per year
     - efficiency_ratio = 1.23^(year - 2022)
     - Example: 2030 chip design would be 1.23^8 ≈ 4.26× more efficient than H100

Source: compute_density/, architecture_efficiency/

================================================================================
LIKELIHOOD RATIO MODEL
================================================================================

The overall likelihood ratio combines three independent intelligence sources:

    LR_total = LR_inventory × LR_procurement × LR_other

Each likelihood ratio represents LR = P(evidence | fab exists) / P(evidence | fab doesn't exist).
Since the three intelligence sources are independent, their likelihood ratios multiply.

This likelihood ratio can later be combined with a prior probability using Bayes' theorem:
    posterior_odds = prior_odds × LR_total

Method 1: Inventory Accounting (LR_inventory)
----------------------------------------------
Compares US intelligence estimates against reported PRC scanner counts, assuming
optimal fab strategy diverts a fixed proportion of scanners.

US intelligence estimates have errors following: P(|error|/actual <= x) = 1 - e^(-kx)
where k is calibrated so P = 0.5 when x = 0.14 (50% chance of 14% or less error).

The PDF of proportional errors is: pdf(error) = k × e^(-k × |error|)

Likelihood ratio calculation compares two hypotheses:
  - Hypothesis 1 (fab exists): True scanners = reported / (1 - diversion_proportion)
    US estimate has some error relative to this true value

  - Hypothesis 2 (no fab): True scanners = reported
    US estimate has some error relative to reported value

  LR_inventory = pdf(us_estimate | fab exists) / pdf(us_estimate | no fab)
               = [k × e^(-k × |us_estimate - true_if_fab| / true_if_fab)] /
                 [k × e^(-k × |us_estimate - reported| / reported)]

Where true_if_fab = reported / (1 - optimal_diversion_proportion)

Interpretation (assuming 10% optimal diversion):
  - If US estimate = reported count → LR ≈ 0.61 (mild evidence against fab)
    * If fab existed, we'd expect US estimate ≈ 11% higher than reported
  - If US estimate = 11% above reported → LR ≈ 1.73 (evidence for fab)
    * US estimate matches expected truth if 10% diverted
  - If US estimate < reported count → LR < 1 (evidence against fab)
    * Contradicts fab hypothesis which would make reported artificially low
  - If US estimate >> reported count → LR increases further
    * Stronger evidence that reported is missing diverted scanners

This represents intelligence agencies comparing their independent estimates against
what PRC reports, considering whether the discrepancy matches expected diversion.

Source: intelligence_error_bars/diversion_vs_detection.py

Method 2: Procurement Accounting (LR_procurement)
-------------------------------------------------
Tracks whether foreign equipment purchases can be accounted for.

The likelihood ratio is binary based on whether PRC achieved localization:
  - If construction starts before localization: Foreign imports required → strong evidence
  - If construction starts after localization: No foreign imports needed → no evidence

Formula:
  - If construction_start < localization_year: LR_procurement = 10 (strong evidence)
  - Otherwise: LR_procurement = 1.0 (no evidence either way)

Interpretation:
  - Without domestic capability, must import from ASML (closely monitored)
  - With domestic capability, can build without triggering procurement alerts
  - Binary outcome: Either clear evidence (LR=10) or no evidence (LR=1)
  - LR=10 represents roughly 10:1 odds in favor of fab existing if foreign imports detected

Localization timing sampled from probability curves by process node:
  - 130nm: ~80% chance by 2026 (mature technology)
  - 28nm: 7% chance by 2028, 25% by 2031
  - 14nm: 3% chance by 2028, 10% by 2031
  - 7nm: 1% chance by 2028, 6% by 2031

Source: china_domestic_fab_capability/forecasts.py

Method 3: Other Intelligence Strategies (LR_other)
--------------------------------------------------
Combines HUMINT (human intelligence), SIGINT (signals intelligence), and IMINT
(imagery intelligence) based on workforce size.

Binary detection model:
  - At initialization, a detection time is sampled from a Gamma distribution
  - The Gamma distribution parameters depend on total labor (construction + operation)
  - Once current time exceeds the sampled detection time, LR jumps to very high value
  - Before detection time, LR remains neutral

Detection time sampling:
  - Mean detection time: μ(workers) = A / log10(workers)^B
    Calibrated from two anchor points:
    * 100 workers → mean detection time of 6.95 years
    * 1000 workers → mean detection time of 3.42 years

  - Variance: σ² = 3.88 (constant uncertainty in detection timing)
  - Gamma parameters: k = μ/σ², θ = σ²
  - Detection time sampled once: t_detect ~ Gamma(k, θ)

Likelihood ratio (using Gamma distribution survival function):
  - If years_since_start ≥ t_detect: LR_other = 100 (strong evidence, project detected)
  - If years_since_start < t_detect: LR_other = SF_Gamma(t)
    * SF_Gamma(t) = P(detection time ≥ t) = 1 - CDF_Gamma(t)
    * This is the survival function of the Gamma distribution
    * LR = P(not detected by t | fab exists) / P(not detected by t | no fab)
         = SF_Gamma(t) / 1.0 = SF_Gamma(t)
    * As time passes, SF decreases, so LR decreases
    * Evidence logic: The longer a fab operates undetected, the less likely it exists
    * This is the principled Bayesian approach using the actual detection time distribution

Interpretation:
  - More workers → shorter mean detection time (sampled once at project start)
    * HUMINT: More potential informants, defectors, and loose talk
    * SIGINT: More communications to intercept
    * IMINT: Larger facilities easier to spot via satellites

  - Detection is binary and irreversible: once detected, always detected
  - Variance captures uncertainty in when the breakthrough intelligence event occurs

  - Example mean detection times:
    * 100 workers → mean 6.95 years
    * 1000 workers → mean 3.42 years
    * 10000 workers → mean ~1.7 years

Calibrated from historical covert nuclear facility detection cases.

Source: detection_probability_vs_num_workers/model_bayesian.py

================================================================================
USAGE
================================================================================

Instantiate PRCCovertFab with construction parameters:
    fab = PRCCovertFab(
        construction_start_year=2026,
        construction_labor=500,
        process_node=ProcessNode.nm28,
        proportion_of_prc_lithography_scanners_devoted_to_fab=0.10,
        operation_labor=2000,
        agreement_year=2030
    )

Then query compute production and likelihood ratio:
    compute_per_month = fab.compute_produced_per_month(year=2030)
    likelihood_ratio = fab.likelihood_ratio_from_evidence(year=2030)
"""


class FabNotBuiltException(Exception):
    """Exception raised when fab cannot be built due to process node threshold not being met"""
    pass

# ============================================================================
# SHARED TYPES AND CONSTANTS
# ============================================================================

# H100 reference constants
H100_PROCESS_NODE_NM = 4  # H100 was fabricated on 4nm node
H100_RELEASE_YEAR = 2022  # H100 was released in 2022
H100_TRANSISTOR_DENSITY_M_PER_MM2 = 98.28  # Million transistors per mm²
H100_WATTS_PER_TPP = 0.326493  # Watts per Tera-Parameter-Pass
H100_TPP_PER_CHIP = 2144.0  # Tera-Parameter-Passes per H100 chip (134 TFLOP/s FP16 * 16 bits)

class ProcessNodeStrategy(Enum):
    """Strategy for selecting process node for covert fab"""
    BEST_INDIGENOUS = "best_indigenous"
    BEST_INDIGENOUS_GTE_28NM = "best_indigenous_gte_28nm"
    BEST_INDIGENOUS_GTE_14NM = "best_indigenous_gte_14nm"
    BEST_INDIGENOUS_GTE_7NM = "best_indigenous_gte_7nm"

# ============================================================================
# Helper functions for estimating fab compute production
# ============================================================================

def estimate_wafer_starts_per_month(
    operation_labor: float,
    number_of_prc_lithography_scanners_devoted_to_fab: float
) -> float:
    """
    Estimates wafer starts per month using fixed-proportions production function with uncertainty.

    Production is limited by the more constraining factor between labor and equipment.
    This represents the reality that you need both workers and scanners to produce wafers.

    Labor productivity has inherent uncertainty (62% relative std) due to variation in:
    - Management efficiency
    - Worker skill levels
    - Process maturity
    - Factory layout and logistics

    Args:
        operation_labor: Number of workers dedicated to fab operation
        number_of_prc_lithography_scanners_devoted_to_fab: Number of lithography scanners allocated to the fab

    Returns:
        float: Estimated wafer starts per month (with random variation)
    """
    # Base labor capacity with uncertainty
    median_labor_capacity = FabModelParameters.wafers_per_month_per_worker * operation_labor

    # Sample labor capacity from a log-normal distribution
    # Median of median_labor_capacity, relative std of 0.62 (from empirical data)
    sigma_relative = FabModelParameters.labor_productivity_relative_sigma
    wafers_per_month_achievable_given_operation_labor = sample_from_log_normal(median_labor_capacity, sigma_relative)

    # Base scanner capacity with uncertainty
    median_scanner_capacity = FabModelParameters.wafers_per_month_per_lithography_scanner * number_of_prc_lithography_scanners_devoted_to_fab

    # Sample scanner capacity from a log-normal distribution (ESTIMATED: 20% relative uncertainty)
    sigma_relative_scanner = FabModelParameters.scanner_productivity_relative_sigma
    wafers_per_month_achievable_given_lithography_scanners = sample_from_log_normal(median_scanner_capacity, sigma_relative_scanner)

    # Return minimum of labor and scanner capacity, with a floor of 1 wafer/month to prevent log(0) errors
    result = min(wafers_per_month_achievable_given_operation_labor, wafers_per_month_achievable_given_lithography_scanners)
    return max(result, 1.0)

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

        median_duration = slope * np.log10(wafer_starts_per_month) + intercept

        # Add uncertainty to construction time (includes both regression and residual uncertainty)
        sigma_relative = FabModelParameters.construction_time_relative_sigma
        return sample_from_log_normal(median_duration, sigma_relative)

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
        H100_PROCESS_NODE_NM / process_node_nm
    ) ** FabModelParameters.transistor_density_scaling_exponent

    return density_relative_to_h100

def estimate_architecture_efficiency_relative_to_h100(year: float, agreement_year: float = None) -> float:
    """
    Estimates chip architecture efficiency relative to H100 based on year.

    Architecture improvements compound at 1.23× per year. The H100 (released in 2022)
    serves as the reference point with efficiency = 1.0.

    If agreement_year is provided, architecture efficiency stops improving after that year
    and remains constant at the agreement year level.

    Args:
        year: The year to calculate architecture efficiency for
        agreement_year: Optional year after which architecture efficiency stops improving

    Returns:
        float: Architecture efficiency relative to H100 (1.0 = H100 in 2022, >1 = better, <1 = worse)
    """
    # If agreement year is specified and we're past it, cap at agreement year efficiency
    if agreement_year is not None and year > agreement_year:
        year = agreement_year

    years_since_h100 = year - H100_RELEASE_YEAR
    efficiency_relative_to_h100 = FabModelParameters.architecture_efficiency_improvement_per_year ** years_since_h100

    return efficiency_relative_to_h100

from functools import lru_cache

@lru_cache(maxsize=None)
def _get_localization_cdf(process_node: ProcessNode) -> tuple:
    """
    Cached helper to compute the CDF for localization years.
    Returns the CDF years and probabilities as tuples (immutable for caching).
    """
    data = FabModelParameters.Probability_of_90p_PRC_localization_at_node[process_node]
    data_years = np.array([point[0] for point in data])
    data_probabilities = np.array([point[1] for point in data])

    # Suppress polyfit warnings
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        coeffs = np.polyfit(data_years, data_probabilities, deg=2)

    prob_curve = np.poly1d(coeffs)

    year_min = data_years.min()
    year_max = data_years.max() + 50
    dense_years = np.linspace(year_min, year_max, 1000)
    dense_probabilities = prob_curve(dense_years)
    dense_probabilities = np.clip(dense_probabilities, 0, 1)

    cdf_probabilities = np.linspace(0, 1, 100)
    cdf_years = np.interp(cdf_probabilities, dense_probabilities, dense_years)

    return tuple(cdf_years), tuple(cdf_probabilities)

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

    # Get cached CDF
    cdf_years_tuple, cdf_probabilities_tuple = _get_localization_cdf(process_node)
    cdf_years = np.array(cdf_years_tuple)
    cdf_probabilities = np.array(cdf_probabilities_tuple)

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

def sample_all_node_localization_years() -> dict:
    """
    Samples localization years for all process nodes at once.

    Ensures consistency: achieving an advanced node means all less advanced nodes are also achieved.
    Node ordering from most to least advanced: nm7, nm14, nm28, nm130

    Returns:
        dict: Mapping from ProcessNode to localization year
    """
    # Order nodes from most advanced to least advanced
    nodes_advanced_to_basic = [ProcessNode.nm7, ProcessNode.nm14, ProcessNode.nm28, ProcessNode.nm130]

    localization_years = {}
    previous_year = float('inf')  # Start with infinity (most advanced node unconstrained)

    for node in nodes_advanced_to_basic:
        # Sample this node's localization year
        sampled_year = sample_year_prc_achieved_node_localization(node)

        # Ensure this less advanced node is achieved no later than more advanced nodes
        # (if you can make 7nm, you can make 14nm)
        actual_year = min(sampled_year, previous_year)
        localization_years[node] = actual_year
        previous_year = actual_year

    return localization_years

def determine_best_available_node(construction_start_year: float, localization_years: dict) -> ProcessNode:
    """
    Determines the best (most advanced) process node available indigenously at construction start.

    Args:
        construction_start_year: Year when fab construction starts
        localization_years: Dict mapping ProcessNode to localization year

    Returns:
        ProcessNode: Best available node, or nm130 if none achieved yet
    """
    # Order nodes from most to least advanced
    nodes_advanced_to_basic = [ProcessNode.nm7, ProcessNode.nm14, ProcessNode.nm28, ProcessNode.nm130]

    for node in nodes_advanced_to_basic:
        if localization_years[node] <= construction_start_year:
            return node

    # If no node achieved yet, default to nm130 (least advanced, most likely to be available)
    return ProcessNode.nm130

def determine_best_available_node_with_min_threshold(
    construction_start_year: float,
    localization_years: dict,
    min_node: ProcessNode
) -> Optional[ProcessNode]:
    """
    Determines the best (most advanced) process node available indigenously at construction start,
    but only if it meets the minimum threshold. Returns None if threshold not met.

    Args:
        construction_start_year: Year when fab construction starts
        localization_years: Dict mapping ProcessNode to localization year
        min_node: Minimum acceptable process node (more advanced or equal)

    Returns:
        ProcessNode: Best available node that meets threshold, or None if threshold not met
    """
    # Order nodes from most to least advanced
    nodes_advanced_to_basic = [ProcessNode.nm7, ProcessNode.nm14, ProcessNode.nm28, ProcessNode.nm130]

    # Find the minimum acceptable nodes (min_node and everything more advanced)
    min_threshold_nm = min_node.to_nm()
    acceptable_nodes = [node for node in nodes_advanced_to_basic if node.to_nm() <= min_threshold_nm]

    # Find best available node among acceptable ones
    for node in acceptable_nodes:
        if localization_years[node] <= construction_start_year:
            return node

    # No acceptable node available yet
    return None

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
    median_total_scanners = (FabModelParameters.prc_lithography_scanners_produced_in_first_year * (n + 1) +
                          FabModelParameters.prc_additional_lithography_scanners_produced_per_year * n * (n + 1) / 2)

    # Add uncertainty to PRC scanner production (ESTIMATED: 30% relative uncertainty)
    sigma_relative = FabModelParameters.prc_scanner_production_relative_sigma
    return sample_from_log_normal(median_total_scanners, sigma_relative)

# ============================================================================
# Helper functions for estimating likelihood ratios
# ============================================================================

def sample_us_estimate_of_prc_scanners(actual_prc_scanners: float) -> float:
    """
    Sample US intelligence estimate of PRC scanner count given actual count.

    US intelligence errors follow CDF: P(|error| / actual <= x) = 1 - e^(-kx)
    where k is calibrated so that P = 0.5 when x = 0.14 (14% error rate).

    This function samples a proportional error from this distribution and applies it
    to the actual count.

    Args:
        actual_prc_scanners: True number of PRC lithography scanners

    Returns:
        float: US intelligence estimate of PRC scanner count
    """
    k = -np.log(0.5) / FabModelParameters.median_absolute_relative_error_of_us_intelligence_estimate_of_prc_sme_stock

    # Sample a uniform random value to invert the CDF
    u = random.random()

    # Invert the CDF: P(|error|/actual <= x) = 1 - e^(-kx)
    # u = 1 - e^(-kx)
    # e^(-kx) = 1 - u
    # -kx = ln(1 - u)
    # x = -ln(1 - u) / k
    proportional_error_magnitude = -np.log(1 - u) / k

    # Randomly choose direction of error (overestimate or underestimate)
    error_sign = 1 if random.random() > 0.5 else -1
    proportional_error = error_sign * proportional_error_magnitude

    # Apply error to actual count
    us_estimate = actual_prc_scanners * (1 + proportional_error)

    # Ensure estimate is non-negative
    return max(0, us_estimate)

def lr_from_inventory_accounting(
    reported_prc_scanners: float,
    us_estimate_of_prc_scanners: float,
    optimal_diversion_proportion: float = 0.1
) -> float:
    """
    Likelihood ratio from inventory accounting evidence - tracking missing SME (photolithography scanners).

    Based on historical intelligence error data showing relationship between proportion of
    diverted equipment and detection probability. Uses bounded exponential model derived from
    intelligence_error_bars/diversion_vs_detection.py.

    The likelihood ratio is computed as: LR = pdf(central_estimate) / pdf(reported_prc_scanners)
    where pdf is the probability density function of the US estimate error distribution.

    Args:
        reported_prc_scanners: What PRC reports (actual scanners minus diverted ones)
        us_estimate_of_prc_scanners: US intelligence estimate of total PRC scanners
        optimal_diversion_proportion: Proportion of scanners diverted to fab (default 0.1 = 10%)

    Returns:
        float: Likelihood ratio LR = P(evidence|fab exists) / P(evidence|fab doesn't exist)
    """

    # Consider two cases:
    # Case 1: Fab exists of optimal size
    # Case 2: Fab does not exist

    # Case 1: Fab exists of optimal size
    # What is the probability that the us_estimate_of_prc_scanners would be what it is in this case?

    true_number_of_prc_scanners_if_fab_exists = reported_prc_scanners / (1 - optimal_diversion_proportion)

    # Handle edge cases where scanner counts are zero or near zero
    if true_number_of_prc_scanners_if_fab_exists < 1e-10 or reported_prc_scanners < 1e-10:
        return 1.0  # Neutral evidence if no scanners

    us_estimate_absolute_error_if_fab_exists = abs(us_estimate_of_prc_scanners - true_number_of_prc_scanners_if_fab_exists) / true_number_of_prc_scanners_if_fab_exists

    # PDF of absolute error
    k = -np.log(0.5) / FabModelParameters.median_absolute_relative_error_of_us_intelligence_estimate_of_prc_sme_stock
    p_observe_us_estimate_error_if_fab_exists = k * np.exp(-k * us_estimate_absolute_error_if_fab_exists)

    # Case 2: Fab does not exist
    true_number_of_prc_scanners_if_no_fab = reported_prc_scanners
    us_estimate_absolute_error_if_no_fab = abs(us_estimate_of_prc_scanners - true_number_of_prc_scanners_if_no_fab) / true_number_of_prc_scanners_if_no_fab
    p_observe_us_estimate_error_if_no_fab = k * np.exp(-k * us_estimate_absolute_error_if_no_fab)

    if p_observe_us_estimate_error_if_no_fab > 0:
        lr = p_observe_us_estimate_error_if_fab_exists / p_observe_us_estimate_error_if_no_fab
    else:
        lr = 1e6  # Very large if no-fab scenario makes estimate very unlikely

    return lr

def lr_from_procurement_accounting(
    construction_start_year: float,
    year_process_node_achieved_90p_prc_localization: float
) -> float:
    """
    Likelihood ratio from procurement accounting evidence - tracking purchases of foreign SME.

    If the fab's process node has achieved >90% domestic localization by construction start,
    then most equipment can be sourced domestically without triggering foreign procurement alerts.
    Otherwise, the need to import foreign SME makes detection highly likely.

    Args:
        construction_start_year: The year when fab construction starts
        year_process_node_achieved_90p_prc_localization: The year when PRC achieved 90% localization for this node

    Returns:
        float: Likelihood ratio (very large if foreign imports required, 1.0 otherwise)
    """
    # Binary outcome: if localization achieved before construction start, no evidence (LR=1)
    # Otherwise, foreign imports required, creating strong evidence (LR very large)
    if year_process_node_achieved_90p_prc_localization > construction_start_year:
        return 10  # Large LR representing high probability of detection via foreign procurement
    else:
        return 1.0  # No evidence from procurement (neutral)

# ============================================================================
# Energy efficiency prediction
# ============================================================================

def predict_watts_per_tpp_from_transistor_density(transistor_density_m_per_mm2: float) -> float:
    """Predict energy consumption (watts per TPP) from transistor density.

    Uses power law relationships that differ before and after Dennard scaling ended (~2006).

    Args:
        transistor_density_m_per_mm2: Transistor density in millions of transistors per mm²

    Returns:
        Watts per Tera-Parameter-Pass (W/TPP)
    """
    # Get parameters from FabModelParameters
    params = FabModelParameters

    # Always calculate post-Dennard line first (anchored to H100)
    # Then calculate pre-Dennard line to connect to it at the transition point

    # Calculate the transition point value using the post-Dennard relationship
    transition_density = params.transistor_density_at_end_of_dennard_scaling_m_per_mm2
    transition_density_ratio = transition_density / H100_TRANSISTOR_DENSITY_M_PER_MM2
    transition_watts_per_tpp = H100_WATTS_PER_TPP * (transition_density_ratio ** params.watts_per_tpp_vs_transistor_density_exponent_after_dennard_scaling_ended)

    # Determine which regime we're in
    if transistor_density_m_per_mm2 < params.transistor_density_at_end_of_dennard_scaling_m_per_mm2:
        # Before Dennard scaling ended - use transition point as anchor
        exponent = params.watts_per_tpp_vs_transistor_density_exponent_before_dennard_scaling_ended
        density_ratio = transistor_density_m_per_mm2 / transition_density
        watts_per_tpp = transition_watts_per_tpp * (density_ratio ** exponent)
    else:
        # After Dennard scaling ended - anchor to H100
        exponent = params.watts_per_tpp_vs_transistor_density_exponent_after_dennard_scaling_ended
        density_ratio = transistor_density_m_per_mm2 / H100_TRANSISTOR_DENSITY_M_PER_MM2
        watts_per_tpp = H100_WATTS_PER_TPP * (density_ratio ** exponent)

    return watts_per_tpp

def process_node_to_nm(process_node: ProcessNode) -> float:
    """Convert ProcessNode enum to nanometer value.

    Args:
        process_node: ProcessNode enum value

    Returns:
        Process node size in nanometers
    """
    node_map = {
        ProcessNode.nm130: 130.0,
        ProcessNode.nm28: 28.0,
        ProcessNode.nm14: 14.0,
        ProcessNode.nm7: 7.0,
    }
    return node_map[process_node]

def calculate_transistor_density_from_process_node(process_node_nm: float) -> float:
    """Calculate transistor density from process node size.

    Uses Moore's Law scaling based on H100 as reference point.

    Args:
        process_node_nm: Process node size in nanometers

    Returns:
        Transistor density in millions of transistors per mm²
    """
    params = FabModelParameters

    # Calculate how many times the process node has been halved relative to H100
    # node_ratio > 1 means larger/older node, node_ratio < 1 means smaller/newer node
    node_ratio = process_node_nm / H100_PROCESS_NODE_NM

    # Transistor density scales as: density = H100_density * (node_ratio)^(-exponent)
    # When node size doubles, density decreases by 2^exponent
    # When node size halves, density increases by 2^exponent
    transistor_density = H100_TRANSISTOR_DENSITY_M_PER_MM2 * (node_ratio ** (-params.transistor_density_scaling_exponent))

    return transistor_density

# ============================================================================
# PRC covert fab core logic
# ============================================================================

class CovertFab(ABC):
    @abstractmethod
    def is_operational(self, year):
        return None

    @abstractmethod
    def compute_produced_per_month(self, year):
        """Return Compute object representing monthly production."""
        return None

    @abstractmethod
    def detection_likelihood_ratio(self, year):
        return None

@dataclass
class PRCCovertFab(CovertFab):
    # Properties that are used to determine compute production
    construction_start_year : float
    construction_duration : float
    wafer_starts_per_month : float
    h100_sized_chips_per_wafer : float
    agreement_year : float

    # Properties that are used to determine detection probability
    process_node : ProcessNode
    year_process_node_achieved_90p_prc_localization : float
    proportion_of_prc_lithography_scanners_devoted_to_fab : float
    construction_labor : int
    operation_labor : int
    us_estimate_of_prc_lithography_scanners : float

    # Fields with default values must come last
    dark_compute_monthly_production_rate_history : dict = field(default_factory=dict)
    detection_updates : dict = field(default_factory=dict)

    def __init__(
            self,
            construction_start_year : float,
            construction_labor : float,
            process_node,  # Can be ProcessNode enum, ProcessNodeStrategy, or legacy "best_available_indigenously" string
            proportion_of_prc_lithography_scanners_devoted_to_fab : float,
            operation_labor : float,
            agreement_year : float,
    ):
        # Sample all node localization years with consistency constraint (needed for strategy-based selection)
        all_localization_years = sample_all_node_localization_years()

        # Handle different process node selection strategies
        if process_node == "best_available_indigenously" or process_node == ProcessNodeStrategy.BEST_INDIGENOUS or process_node == "best_indigenous":
            # Original behavior: select best available node without threshold
            actual_process_node = determine_best_available_node(construction_start_year, all_localization_years)
            self.year_process_node_achieved_90p_prc_localization = all_localization_years[actual_process_node]
            self.process_node = actual_process_node

        elif process_node == ProcessNodeStrategy.BEST_INDIGENOUS_GTE_28NM or process_node == "best_indigenous_gte_28nm":
            # Only build if 28nm or better is available
            actual_process_node = determine_best_available_node_with_min_threshold(
                construction_start_year, all_localization_years, ProcessNode.nm28
            )
            if actual_process_node is None:
                raise FabNotBuiltException("28nm or better not available indigenously")
            self.year_process_node_achieved_90p_prc_localization = all_localization_years[actual_process_node]
            self.process_node = actual_process_node

        elif process_node == ProcessNodeStrategy.BEST_INDIGENOUS_GTE_14NM or process_node == "best_indigenous_gte_14nm":
            # Only build if 14nm or better is available
            actual_process_node = determine_best_available_node_with_min_threshold(
                construction_start_year, all_localization_years, ProcessNode.nm14
            )
            if actual_process_node is None:
                raise FabNotBuiltException("14nm or better not available indigenously")
            self.year_process_node_achieved_90p_prc_localization = all_localization_years[actual_process_node]
            self.process_node = actual_process_node

        elif process_node == ProcessNodeStrategy.BEST_INDIGENOUS_GTE_7NM or process_node == "best_indigenous_gte_7nm":
            # Only build if 7nm or better is available
            actual_process_node = determine_best_available_node_with_min_threshold(
                construction_start_year, all_localization_years, ProcessNode.nm7
            )
            if actual_process_node is None:
                raise FabNotBuiltException("7nm or better not available indigenously")
            self.year_process_node_achieved_90p_prc_localization = all_localization_years[actual_process_node]
            self.process_node = actual_process_node

        else:
            # Normal case: specific process node provided
            # Sample the year when this process node achieved 90% PRC localization
            # This is used consistently for both detection probability and machine count calculations
            self.year_process_node_achieved_90p_prc_localization = sample_year_prc_achieved_node_localization(process_node)
            self.process_node = process_node

        # Estimate total PRC lithography scanners for this node at construction start
        self.total_prc_lithography_scanners_for_node = estimate_total_prc_lithography_scanners_for_node(
            current_year=construction_start_year,
            localization_year=self.year_process_node_achieved_90p_prc_localization
        )

        # Properties that are used to determine compute production
        self.construction_start_year = construction_start_year
        self.agreement_year = agreement_year

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
        self.transistor_density_relative_to_h100 = estimate_transistor_density_relative_to_h100(self.process_node.to_nm())

        # Production capacity (H100e per month, regardless of operational status)
        # This represents the potential production rate when the fab is operational
        self.production_capacity = self.wafer_starts_per_month * self.h100_sized_chips_per_wafer * self.transistor_density_relative_to_h100

        # Properties that are used to determine likelihood ratio from covert project evidence
        # Note: self.process_node already set above based on input
        self.proportion_of_prc_lithography_scanners_devoted_to_fab = proportion_of_prc_lithography_scanners_devoted_to_fab
        self.construction_labor = construction_labor
        self.operation_labor = operation_labor
        self.us_estimate_of_prc_lithography_scanners = sample_us_estimate_of_prc_scanners(
            self.total_prc_lithography_scanners_for_node
        )
        # Initialize fields that have default_factory in dataclass
        self.dark_compute_monthly_production_rate_history = {}
        self.detection_updates = {}

        self.lr_inventory = lr_from_inventory_accounting(
            reported_prc_scanners= (1 - self.proportion_of_prc_lithography_scanners_devoted_to_fab) * self.total_prc_lithography_scanners_for_node,
            us_estimate_of_prc_scanners=self.us_estimate_of_prc_lithography_scanners,
            optimal_diversion_proportion=self.proportion_of_prc_lithography_scanners_devoted_to_fab
        )

        # Procurement accounting: Determine whether the SME China has imported can be accounted for
        # If PRC hasn't achieved domestic localization, foreign scanner imports would be detectable
        self.lr_procurement = lr_from_procurement_accounting(
            construction_start_year=self.construction_start_year,
            year_process_node_achieved_90p_prc_localization=self.year_process_node_achieved_90p_prc_localization
        )

        self.lr_other_over_time = {}

    def is_operational(self, year):
        if self.construction_start_year != None: # Construction start year can be none if the PRC never builds a covert fab
            return year >= self.construction_start_year + self.construction_duration
        else:
            return False

    def compute_produced_per_month(self, year):
        """Calculate monthly compute production as a Compute object.

        Returns:
            Compute: Object containing chip specifications and counts
        """
        if not self.is_operational(year):
            # Return empty Compute object if not operational
            return Compute(chip_counts={})

        # Calculate H100-equivalent performance per chip
        chip_architecture_efficiency_relative_to_h100 = estimate_architecture_efficiency_relative_to_h100(year, self.agreement_year)
        h100e_per_chip = self.transistor_density_relative_to_h100 * chip_architecture_efficiency_relative_to_h100

        # Convert process node to nm and calculate transistor density
        process_node_nm = process_node_to_nm(self.process_node)
        transistor_density = calculate_transistor_density_from_process_node(process_node_nm)

        # Calculate energy consumption per chip
        # First convert H100e to actual TPP
        tpp_per_chip = h100e_per_chip * H100_TPP_PER_CHIP

        # Get watts per TPP based on transistor density
        watts_per_tpp = predict_watts_per_tpp_from_transistor_density(transistor_density)

        # Calculate total watts per chip
        watts_per_chip = tpp_per_chip * watts_per_tpp

        # Calculate number of chips produced per month
        chips_per_month = self.wafer_starts_per_month * self.h100_sized_chips_per_wafer

        # Create Chip object
        chip = Chip(
            h100e_tpp_per_chip=h100e_per_chip,
            W_of_energy_consumed=watts_per_chip,
            intra_chip_memory_bandwidth_tbps=8.0,  # Default H100-like bandwidth
            inter_chip_memory_bandwidth_tbps=1.8   # Default H100-like bandwidth
        )

        compute_production_rate = Compute(chip_counts={chip: chips_per_month})
        self.dark_compute_monthly_production_rate_history[year] = Compute(chip_counts={chip: chips_per_month})
        return compute_production_rate
    
    def get_cumulative_compute_production(self, year):
        """Calculate cumulative compute production up to a given year.

        Returns:
            Compute: Object containing cumulative chip specifications and counts
        """
        chip_type = self.dark_compute_monthly_production_rate_history[year].chip_counts.keys()[0]
        cumulative_chip_count = 0
        years_recorded = sorted(self.dark_compute_monthly_production_rate_history.keys())
        years_before_year = [yr for yr in years_recorded if yr < year]
        for i, yr in enumerate(years_before_year):
            assert len(self.dark_compute_monthly_production_rate_history[yr].chip_counts) == 1, "Expected only one chip type in production rate history"
            assert chip_type == list(self.dark_compute_monthly_production_rate_history[yr].chip_counts.keys())[0], "Inconsistent chip types in production rate history"
            monthly_chip_count_rate = self.dark_compute_monthly_production_rate_history[yr][chip_type]
            year_increment = years_before_year[i + 1] - yr if i + 1 < len(years_before_year) else year - yr
            months_rate_is_sustained = year_increment * 12
            cumulative_chip_count += months_rate_is_sustained * monthly_chip_count_rate
        return Compute(chip_counts={chip_type: cumulative_chip_count})

    def energy_consumption_per_month_gw(self, year):
        """Calculate total energy consumption per month in gigawatts.

        Returns:
            float: Energy consumption in GW per month
        """
        compute = self.compute_produced_per_month(year)
        return compute.total_energy_requirements_GW()

    def detection_likelihood_ratio(self, year):
        """
        Overall likelihood ratio from intelligence evidence about the fab.

        Combines three independent intelligence sources:
        1. Inventory accounting (missing domestic SME)
        2. Procurement accounting (foreign SME purchases)
        3. Other strategies (HUMINT, SIGINT based on workers and time)

        Multiplies likelihood ratios: LR_total = LR_inventory × LR_procurement × LR_other

        This likelihood ratio can be combined with a prior using Bayes' theorem:
        posterior_odds = prior_odds × LR_total

        Args:
            year: Current year to evaluate likelihood ratio

        Returns:
            float: Overall likelihood ratio from all intelligence sources
        """
        # Inventory accounting: Determine whether Chinese SME has gone missing
        # Tracks whether domestic PRC lithography scanners are unaccounted for

        # Other: Leverage signals intelligence, human intelligence, and satellite imagery to detect a chip fab operation
        # Uses Gamma distribution survival function for principled likelihood ratio
        lr_other = lr_vs_num_workers(
            years_since_start = year - self.construction_start_year,
            total_labor=self.operation_labor + self.construction_labor,
            mean_detection_time_100_workers=FabModelParameters.mean_detection_time_for_100_workers,
            mean_detection_time_1000_workers=FabModelParameters.mean_detection_time_for_1000_workers,
            variance_theta=FabModelParameters.variance_of_detection_time_given_num_workers
        )

        # Store component LRs for tracking over time
        self.lr_other_over_time[year] = lr_other

        return self.lr_inventory * self.lr_procurement * lr_other

    @staticmethod
    def watts_per_tpp_relative_to_H100():
        """Generate the watts per TPP vs transistor density curve data for plotting.

        Returns:
            dict: Contains 'density_relative' and 'watts_per_tpp_relative' arrays
                 for plotting the relationship between transistor density and power efficiency
        """

        # Generate density range in log space (0.001x to 10x H100)
        num_points = 100
        min_density_relative = 0.001
        max_density_relative = 10

        density_relative = np.logspace(np.log10(min_density_relative), np.log10(max_density_relative), num_points)
        density_absolute = density_relative * H100_TRANSISTOR_DENSITY_M_PER_MM2

        # Calculate watts per TPP for each density using the Python function
        watts_per_tpp_absolute = [predict_watts_per_tpp_from_transistor_density(d) for d in density_absolute]

        # Convert to relative values (relative to H100)
        watts_per_tpp_relative = [w / H100_WATTS_PER_TPP for w in watts_per_tpp_absolute]

        return {
            "density_relative": density_relative.tolist(),
            "watts_per_tpp_relative": watts_per_tpp_relative
        }