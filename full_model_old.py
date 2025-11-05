"""
FULL MODEL: Covert Compute Production Forecasting Model
========================================================

This model estimates compute production and detection risk for a covert semiconductor
fab project.

MODEL INPUTS:
-------------
1. agreement_start_year: The year the covert agreement/construction begins
2. year: The year for which to determine the cumulative detection probability, and annual compute production. 
3. num_workers: How many workers are involved in the project

MODEL OUTPUTS:
--------------
1. cumulative_detection_probability: Probability that the project has been
   detected by the input "Year." 
2. annual_compute_production: Distribution of H100-equivalent compute production during "Year."

MODEL FORMULA:
-----------------------
The following is the formula for computing compute production for an individual rollout:

annual_compute_production =
    finished_construction(year - agreement_start_year)
    * wafers_processed_per_month(num_workers)
    * chip_efficiency(year - 2022) / chip_efficiency(2022)
    * H100s_per_wafer_with_2022_architectures(max_selfsufficient_node(agreement_start_year))

cumulative_detection_probability = p_detected_after_years(year - agreement_start_year, num_workers)

COMPONENT MODELS:
-----------------

1. FINISHED_CONSTRUCTION (year, year_of_agreement_start):
   - Construction time ~ Gamma distribution fitted to historical fab construction data
   - Data from 9 fabs, adjusted by 1.5x multiplier for covert construction difficulty
   - Returns 1 if construction finished by evaluation year, 0 otherwise
   - Gamma parameters: shape=4.69, scale=0.98 (from fitted data)

2. WAFERS_PROCESSED_PER_MONTH (num_people_involved):
   - Based on log-linear relationship: log(wafers) = a + b*log(employees)
   - Fitted from 11 fab operating data points
   - Adds log-normal noise (~20% CV) to account for variation
   - Relationship: wafers ≈ employees^1.09 * 10^0.57

3. CHIP_EFFICIENCY (year):
   - Annual improvement multiplier: 1.23x per year
   - Baseline: H100 in 2023 with 63,328 TPP per die
   - efficiency(year) = 63,328 * 1.23^(year - 2023)
   - Ratio used: efficiency(year) / efficiency(2022)

4. MAX_SELFSUFFICIENT_NODE (year_of_agreement_start):
   - Based on China's semiconductor equipment localization forecasts
   - Forecasts give P(node has >90% localization) for 28nm, 14nm, and 7nm

   IMPORTANT - Probability Disentanglement:
   The raw forecasts represent cumulative probabilities:
   - p_28nm_raw = P(best node is 7nm OR 14nm OR 28nm)
   - p_14nm_raw = P(best node is 7nm OR 14nm)
   - p_7nm_raw = P(best node is 7nm)

   We disentangle to get individual node probabilities:
   - P(best node is 7nm) = p_7nm_raw
   - P(best node is 14nm) = p_14nm_raw - p_7nm_raw
   - P(best node is 28nm) = p_28nm_raw - p_14nm_raw
   - P(best node is 120nm) = 1 - p_28nm_raw (fallback if none achieve >90%)

   Sample the best available node (smaller nm = more advanced).
   If none achieve >90% localization, fall back to 120nm.

5. H100S_PER_WAFER (process_node):
   - Based on transistor density at each process node
   - Assumes 2022 H100-like architecture (814mm² die size)
   - Dies per wafer: 28 (from NVIDIA production estimates)
   - TPP per node = transistor_density * 644 (TPP per MTr/mm²)
   - H100_equiv = (28 dies * TPP_node) / 63,328

6. DETECTION MODEL (number_of_people_involved, years_elapsed):
   - Based on Bayesian analysis of 13 covert nuclear programs
   - Detection time ~ Gamma(mean=μ(x), variance=σ²*μ(x))
   - Mean detection time: μ(x) = 9.856 / log10(x)^1.451
   - Variance parameter: σ² = 5.892
   - P(detected by time t) = Gamma_CDF(t; k=μ/σ², scale=σ²)

DATA SOURCES:
-------------
- Fab construction times: fab_construction_time/fab_construction_time.py
- Labor vs production: fab_production_vs_labor_inputs/labor_vs_fab_production.py
- Architecture efficiency: architecture_efficiency/architecture_efficiency.py
- Node capabilities: compute_vs_node/compute_vs_node_multiple_lines.py
- China localization: china_domestic_fab_capability/forecasts.py
- Detection model: detection_probability_vs_num_workers/model_bayesian.py
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ============================================================================
# Constants and Data from other files
# ============================================================================

# From compute_vs_node/compute_vs_node_multiple_lines.py
h100_tpp = 63328
TPP_per_transistor_density = 644
dies_per_wafer = 28

process_node = [3, 4, 5, 6, 7, 10, 12, 16, 22, 28, 40, 65, 90, 120, 130, 180]
transistor_density = [215.6, 145.86, 137.6, 106.96, 90.64, 60.3, 33.8, 28.88, 16.50, 14.44, 7.22, 3.61, 1.80, 0.67, 0.90, 0.45]  # 120nm interpolated between 90nm and 130nm
year_process_node_first_reached_high_volume_manufacturing = [
    2023,  # 3nm
    2022,  # 4nm
    2020,  # 5nm
    2020,  # 6nm
    2018,  # 7nm
    2017,  # 10nm
    2018,  # 12nm
    2015,  # 16nm
    2012,  # 22nm
    2011,  # 28nm
    2009,  # 40nm
    2006,  # 65nm
    2004,  # 90nm
    2001,  # 130nm
    1999   # 180nm
]

# From architecture_efficiency/architecture_efficiency.py
h100_tpp_per_die_size = 63328
h100_release_year = 2023
annual_efficiency_multiplier = 1.23

# From fab_construction_time/construction_time_cdf.py
construction_months = [12, 42, 26, 36, 42, 36, 30, 24, 26]
construction_months_adjusted = [m * 1.5 for m in construction_months]
construction_years_adjusted = [m / 12 for m in construction_months_adjusted]

# Fit gamma distribution for construction time
gamma_shape, gamma_loc, gamma_scale = stats.gamma.fit(construction_years_adjusted, floc=0)

# From fab_production_vs_labor_inputs/labor_vs_fab_production.py
# Operating employees vs wafers per month (for reference)
employees = [80, 1500, 3950, 31000, 11300, 2200, 2750, 3000, 3000, 8500, 10000]
wafers_per_month = [4000, 55000, 31500, 800000, 83000, 20000, 33000, 62500, 100000, 140000, 450000]

# Fit FORWARD relationship: log(wafers) = a + b * log(employees)
log_employees = np.log10(employees)
log_wafers = np.log10(wafers_per_month)
coeffs = np.polyfit(log_employees, log_wafers, 1)
# This gives: log(wafers) = coeffs[1] + coeffs[0] * log(employees)
# Which means: wafers = 10^(coeffs[1]) * employees^(coeffs[0])

# From china_domestic_fab_capability/forecasts.py
p_selfsufficiency_28nm = [{"year": 2025, "probability": 0}, {"year": 2027, "probability": 0.07}, {"year": 2030, "probability": 0.25}]
p_selfsufficiency_14nm = [{"year": 2025, "probability": 0}, {"year": 2027, "probability": 0.03}, {"year": 2030, "probability": 0.1}]
p_selfsufficiency_7nm = [{"year": 2025, "probability": 0}, {"year": 2027, "probability": 0.01}, {"year": 2030, "probability": 0.06}]

# From detection_probability_vs_num_workers/model_bayesian.py
# Bayesian posterior mean parameters for detection model
# μ(x) = A / log10(x)^B where x = number of workers
# Detection time ~ Gamma(mean=μ(x), variance=σ²*μ(x))
# These are the posterior mean values from the MCMC fit
DETECTION_A_MEAN = 9.856  # From posterior mean
DETECTION_B_MEAN = 1.451  # From posterior mean
DETECTION_SIGMA_SQ_MEAN = 5.892  # From posterior mean

# ============================================================================
# Helper Functions
# ============================================================================

def calculate_chip_efficiency(year):
    """Calculate TPP per die size based on years since H100 release (2023)."""
    years_since_2023 = year - h100_release_year
    return h100_tpp_per_die_size * (annual_efficiency_multiplier ** years_since_2023)

def sample_max_selfsufficient_node(year):
    return 130 # (temporary, for testing)
    """
    Sample the maximum self-sufficient node that China can achieve by a given year.
    Returns the process node size in nm.

    The forecasts give cumulative probabilities that need to be disentangled:
    - p_28nm_raw = P(best node is 7nm OR 14nm OR 28nm)
    - p_14nm_raw = P(best node is 7nm OR 14nm)
    - p_7nm_raw = P(best node is 7nm)

    We disentangle to individual probabilities:
    - P(7nm) = p_7nm_raw
    - P(14nm) = p_14nm_raw - p_7nm_raw
    - P(28nm) = p_28nm_raw - p_14nm_raw
    - P(130nm) = 1 - p_28nm_raw (fallback)
    """
    # Find the probabilities at the given year by interpolating
    def interpolate_probability(p_data, year):
        if year <= p_data[0]["year"]:
            return p_data[0]["probability"]
        if year >= p_data[-1]["year"]:
            return p_data[-1]["probability"]

        # Linear interpolation between data points
        for i in range(len(p_data) - 1):
            if p_data[i]["year"] <= year <= p_data[i+1]["year"]:
                t = (year - p_data[i]["year"]) / (p_data[i+1]["year"] - p_data[i]["year"])
                return p_data[i]["probability"] * (1 - t) + p_data[i+1]["probability"] * t
        return 0

    # Get raw cumulative probabilities
    p_28nm_raw = interpolate_probability(p_selfsufficiency_28nm, year)
    p_14nm_raw = interpolate_probability(p_selfsufficiency_14nm, year)
    p_7nm_raw = interpolate_probability(p_selfsufficiency_7nm, year)

    # Disentangle to individual probabilities
    p_7nm = p_7nm_raw
    p_14nm = max(0, p_14nm_raw - p_7nm_raw)
    p_28nm = max(0, p_28nm_raw - p_14nm_raw)
    p_130nm = max(0, 1 - p_28nm_raw)

    # Sample from categorical distribution
    rand = np.random.random()
    if rand < p_7nm:
        return 7
    elif rand < p_7nm + p_14nm:
        return 14
    elif rand < p_7nm + p_14nm + p_28nm:
        return 28
    else:
        return 130  # Fallback to 130nm if none achieve >90% localization

def get_H100s_per_wafer(process_node_nm):
    """
    Calculate H100 equivalents per wafer for a given process node,
    assuming 2022 architecture (H100-like design).
    """
    if process_node_nm not in process_node:
        # Find closest node
        closest_idx = np.argmin([abs(n - process_node_nm) for n in process_node])
        process_node_nm = process_node[closest_idx]

    node_idx = process_node.index(process_node_nm)
    tpp_for_node = transistor_density[node_idx] * TPP_per_transistor_density
    h100_equiv_per_wafer = (dies_per_wafer * tpp_for_node) / h100_tpp
    return h100_equiv_per_wafer

def sample_finished_construction_year(year_of_agreement_start):
    """
    Sample when construction finishes given the year the agreement starts.
    Returns the year construction is finished.
    """
    construction_time_years = stats.gamma.rvs(gamma_shape, loc=gamma_loc, scale=gamma_scale)
    return year_of_agreement_start + construction_time_years

def sample_wafers_processed_per_month(num_people_involved):
    """
    Sample wafers processed per month given number of people involved.
    Uses the forward log-linear relationship from labor vs fab production data.

    Adds uncertainty around the fitted relationship.
    """
    # Forward relationship: log(wafers) = coeffs[1] + coeffs[0] * log(employees)
    log_expected_wafers = coeffs[1] + coeffs[0] * np.log10(num_people_involved)
    expected_wafers = 10 ** log_expected_wafers

    # Add log-normal noise (assume ~20% coefficient of variation)
    sigma = 0.2  # This creates roughly 20% CV in linear space
    actual_wafers = expected_wafers * np.random.lognormal(0, sigma)

    return actual_wafers

def finished_construction(year, year_of_agreement_start, construction_finish_year):
    """
    Returns 1 if construction is finished by the given year, 0 otherwise.
    """
    return 1.0 if year >= construction_finish_year else 0.0

def detection_mean_time(num_workers):
    """
    Calculate the mean detection time in years for a given number of workers.
    Uses the Bayesian posterior mean: μ(x) = A / log10(x)^B
    """
    if num_workers <= 1:
        return np.inf  # Very few workers, practically undetectable
    return DETECTION_A_MEAN / (np.log10(num_workers) ** DETECTION_B_MEAN)

def sample_detection_year(year_of_agreement_start, num_workers):
    """
    Sample the year when the project is detected.
    Returns the year of detection, or np.inf if never detected (within reasonable time).

    Detection time follows: Gamma(mean=μ(x), variance=σ²*μ(x))
    where μ(x) = A / log10(x)^B
    """
    if num_workers <= 1:
        return np.inf

    mu = detection_mean_time(num_workers)
    sigma_sq = DETECTION_SIGMA_SQ_MEAN

    # Gamma parameters: mean = k*θ, variance = k*θ²
    # Given variance = σ² * μ, we have: k*θ² = σ²*μ
    # Combined with k*θ = μ, we get: θ = σ², k = μ/σ²
    theta = sigma_sq
    k = mu / sigma_sq

    if k <= 0 or theta <= 0:
        return np.inf

    # Sample detection time (years after agreement start)
    detection_time_years = stats.gamma.rvs(k, scale=theta)

    return year_of_agreement_start + detection_time_years

def calculate_detection_probability(years_elapsed, num_workers):
    """
    Calculate the probability of detection by a given number of years after project start.

    Args:
        years_elapsed: Number of years since the agreement/project start
        num_workers: Number of workers involved

    Returns:
        Probability of detection (0 to 1)
    """
    if num_workers <= 1 or years_elapsed <= 0:
        return 0.0

    mu = detection_mean_time(num_workers)
    sigma_sq = DETECTION_SIGMA_SQ_MEAN

    # Gamma parameters
    theta = sigma_sq
    k = mu / sigma_sq

    if k <= 0 or theta <= 0:
        return 0.0

    # CDF of gamma distribution gives P(detected by time t)
    return stats.gamma.cdf(years_elapsed, a=k, scale=theta)

# ============================================================================
# Main Model Functions
# ============================================================================

def rollout(year, year_of_agreement_start, number_of_people_involved):
    """
    Single Monte Carlo rollout of compute production.

    Computes:
    compute_production(year, year_of_agreement_start, number_of_people_involved) =
        finished_construction(year, year_of_agreement_start)
        * wafers_processed_per_month(num_people_involved)
        * chip_efficiency(year) / chip_efficiency(2022)
        * H100s_per_wafer_with_2022_architectures(max_selfsufficient_node(year_of_agreement_start))

    Returns:
        dict: Contains:
            - h100_equivalents_per_year: Total H100 equivalents produced per year
            - construction_finish_year: Year when construction finishes
            - wafers_per_month: Wafers processed per month
            - max_node: Maximum self-sufficient node achieved
            - is_operational: Whether the fab is operational in the given year
    """

    # Sample construction finish year
    construction_finish_year = sample_finished_construction_year(year_of_agreement_start)

    # Check if construction is finished
    is_operational = finished_construction(year, year_of_agreement_start, construction_finish_year)

    if not is_operational:
        return {
            'h100_equivalents_per_year': 0,
            'construction_finish_year': construction_finish_year,
            'wafers_per_month': 0,
            'max_node': None,
            'is_operational': False,
        }

    # Sample wafers processed per month
    wafers_per_month_value = sample_wafers_processed_per_month(number_of_people_involved)

    # Sample max self-sufficient node (based on year_of_agreement_start)
    max_node = sample_max_selfsufficient_node(year_of_agreement_start)

    # Calculate chip efficiency ratio (relative to 2022)
    chip_eff_year = calculate_chip_efficiency(year)
    chip_eff_2022 = calculate_chip_efficiency(2022)
    efficiency_ratio = chip_eff_year / chip_eff_2022

    # Get H100s per wafer for the max self-sufficient node
    h100s_per_wafer = get_H100s_per_wafer(max_node)

    # Calculate total compute production
    h100_equiv_per_year = (
        is_operational
        * wafers_per_month_value
        * 12  # months per year
        * efficiency_ratio
        * h100s_per_wafer
    )

    return {
        'h100_equivalents_per_year': h100_equiv_per_year,
        'construction_finish_year': construction_finish_year,
        'wafers_per_month': wafers_per_month_value,
        'max_node': max_node,
        'is_operational': True,
    }

def model(agreement_start_year, year, num_workers, num_samples=10000):
    """
    Main model function that produces detection probability and compute production distribution.

    Args:
        agreement_start_year: The year the covert agreement/construction begins
        year: The year for which to determine the cumulative detection probability
              and annual compute production
        num_workers: How many workers are involved in the project
        num_samples: Number of Monte Carlo samples to run (default: 10000)

    Returns:
        dict: Contains two main outputs:
            cumulative_detection_probability: Cumulative probability that the project
                                             has been detected by the input year (scalar, 0 to 1)
            annual_compute_production: Distribution of H100-equivalent compute production
                                       during the input year
                - h100_equivalents_per_year: Array of sampled values
                - percentiles: Dict with p10, p25, p50, p75, p90
                - mean: Mean value
                - std: Standard deviation

            Additional diagnostic information:
                - years_into_agreement: How many years have passed (year - agreement_start_year)
                - operational_probability: Probability fab is operational and construction complete
                - construction_finish_years: Array of sampled construction finish years
    """
    # Calculate years into agreement
    years_into_agreement = year - agreement_start_year

    # Calculate cumulative detection probability directly using gamma CDF
    # P(detected by year) = P(detection_time <= years_into_agreement)
    cumulative_detection_probability = calculate_detection_probability(years_into_agreement, num_workers)

    # Run Monte Carlo samples for compute production
    h100_samples = []
    construction_years = []
    wafer_samples = []
    node_samples = []
    operational_count = 0

    for _ in range(num_samples):
        result = rollout(year, agreement_start_year, num_workers)
        h100_samples.append(result['h100_equivalents_per_year'])
        construction_years.append(result['construction_finish_year'])
        wafer_samples.append(result['wafers_per_month'])
        if result['max_node'] is not None:
            node_samples.append(result['max_node'])
        if result['is_operational']:
            operational_count += 1

    h100_array = np.array(h100_samples)

    return {
        # PRIMARY OUTPUTS
        'cumulative_detection_probability': cumulative_detection_probability,
        'annual_compute_production': {
            'h100_equivalents_per_year': h100_array,
            'percentiles': {
                'p10': np.percentile(h100_array, 10),
                'p25': np.percentile(h100_array, 25),
                'p50': np.percentile(h100_array, 50),
                'p75': np.percentile(h100_array, 75),
                'p90': np.percentile(h100_array, 90),
            },
            'mean': np.mean(h100_array),
            'std': np.std(h100_array),
        },

        # DIAGNOSTIC INFORMATION
        'years_into_agreement': years_into_agreement,
        'operational_probability': operational_count / num_samples,
        'construction_finish_years': np.array(construction_years),
        'wafers_per_month': np.array(wafer_samples),
        'max_nodes': np.array(node_samples),
    }

def compute_produced_before_detection(agreement_start_year, num_workers, num_samples=10000,
                                     max_years=15, output_filename='compute_before_detection.png'):
    """
    Calculate the distribution of total compute produced before the project is detected.

    This function runs Monte Carlo simulations where for each sample:
    1. Sample a detection year from the detection time distribution
    2. Sum up compute production from agreement start until detection
    3. Return distribution of total compute produced before detection

    Args:
        agreement_start_year: The year the covert agreement/construction begins
        num_workers: How many workers are involved in the project
        num_samples: Number of Monte Carlo samples to run (default: 10000)
        max_years: Maximum number of years to simulate (default: 15)
        output_filename: Filename for the output plot (default: 'compute_before_detection.png')

    Returns:
        dict: Contains:
            - total_compute_before_detection: Array of total H100 equivalents produced
            - detection_years: Array of sampled detection years
            - percentiles: Dict with p10, p25, p50, p75, p90
            - mean: Mean total H100 equivalents produced
            - std: Standard deviation
            - output_file: Path to saved visualization
    """
    total_compute_samples = []
    detection_year_samples = []
    operational_time_samples = []

    for _ in range(num_samples):
        # Sample the "world" parameters once for this Monte Carlo sample
        # 1. When does detection occur?
        detection_time_years = sample_detection_year(agreement_start_year, num_workers) - agreement_start_year

        # 2. When does construction finish?
        construction_finish_year = sample_finished_construction_year(agreement_start_year)
        construction_time_years = construction_finish_year - agreement_start_year

        # 3. What process node is achieved?
        max_node = sample_max_selfsufficient_node(agreement_start_year)

        # 4. What is the wafer production capacity?
        wafers_per_month = sample_wafers_processed_per_month(num_workers)

        # Calculate operational time (lower bounded at 0)
        operational_time = max(0, detection_time_years - construction_time_years)
        operational_time_samples.append(operational_time)

        # Cap detection time at max_years for practical purposes
        detection_time_years = min(detection_time_years, max_years)

        # Sum compute production using weekly intervals (1/4 month) for smoother distribution
        # Now we use the same world parameters throughout
        total_compute = 0
        num_intervals = int(detection_time_years * 48)  # Convert years to quarter-months (weeks)

        h100s_per_wafer = get_H100s_per_wafer(max_node)

        for interval_offset in range(num_intervals):
            current_year = agreement_start_year + interval_offset / 48.0

            # Check if construction is finished at this time
            is_operational = (current_year >= construction_finish_year)

            if is_operational:
                # Calculate chip efficiency ratio (relative to 2022)
                chip_eff_year = calculate_chip_efficiency(current_year)
                chip_eff_2022 = calculate_chip_efficiency(2022)
                efficiency_ratio = chip_eff_year / chip_eff_2022

                # Calculate compute for this time interval
                h100_equiv_per_year = (
                    wafers_per_month
                    * 12  # months per year
                    * efficiency_ratio
                    * h100s_per_wafer
                )

                # Add 1/48 of the annual production for this quarter-month
                total_compute += h100_equiv_per_year / 48.0

        total_compute_samples.append(total_compute)
        detection_year_samples.append(agreement_start_year + detection_time_years)

    total_compute_array = np.array(total_compute_samples)
    operational_time_array = np.array(operational_time_samples)

    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Histogram of total compute produced (filter out zeros for log scale)
    nonzero_compute = total_compute_array[total_compute_array > 0]
    if len(nonzero_compute) > 0:
        # Use log-spaced bins (25 bins for smoother appearance, 2x larger than before)
        bins = np.logspace(np.log10(max(0.1, nonzero_compute.min())),
                          np.log10(nonzero_compute.max()), 25)
        ax1.hist(nonzero_compute, bins=bins, alpha=0.7, color='blue', edgecolor='black', density=True)
    ax1.set_xlabel('Total H100 equivalents produced before detection', fontsize=11)
    ax1.set_ylabel('Probability Density', fontsize=11)
    ax1.set_title(f'Distribution of Compute Produced Before Detection\n({num_workers:,} workers, non-zero values only)',
                  fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Complementary CDF (survival function) - P(compute > x)
    # Plot complementary CDF for all values
    sorted_compute = np.sort(total_compute_array)
    # Complementary CDF: P(X > x) = 1 - P(X <= x)
    ccdf = 1 - np.arange(1, len(sorted_compute) + 1) / len(sorted_compute)

    # Only plot values >= 1000 (10^3)
    mask = sorted_compute >= 1000
    if np.any(mask):
        # Draw horizontal line from y-axis to first point
        first_x = sorted_compute[mask][0]
        first_y = ccdf[mask][0]
        ax2.plot([1000, first_x], [first_y, first_y], 'b-', linewidth=2)
        # Plot the complementary CDF
        ax2.plot(sorted_compute[mask], ccdf[mask], linewidth=2, color='blue')

    ax2.set_xlabel('Total H100 equivalents produced before detection', fontsize=11)
    ax2.set_ylabel('P(Compute Produced > x)', fontsize=11)
    ax2.set_title('Complementary CDF of Compute Produced Before Detection', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_xlim(left=1000)  # Start x-axis at 10^3
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    # Add percentile annotations (only for nonzero values >= 1000)
    for percentile in [10, 25, 50, 75, 90]:
        value = np.percentile(total_compute_array, percentile)
        if value >= 1000:  # Only annotate if value is >= 1000
            # For complementary CDF, the probability is 1 - percentile/100
            prob = 1 - percentile/100
            ax2.axhline(prob, color='gray', linestyle=':', alpha=0.5)
            ax2.axvline(value, color='gray', linestyle=':', alpha=0.5)
            ax2.text(value, prob + 0.02, f'p{percentile}', fontsize=8)

    # Plot 3: Complementary CDF of operational time before detection
    sorted_op_time = np.sort(operational_time_array)
    # Complementary CDF: P(X > x) = 1 - P(X <= x)
    ccdf_op = 1 - np.arange(1, len(sorted_op_time) + 1) / len(sorted_op_time)

    ax3.plot(sorted_op_time, ccdf_op, linewidth=2, color='green')

    ax3.set_xlabel('Operational time before detection (years)', fontsize=11)
    ax3.set_ylabel('P(Operational time > x)', fontsize=11)
    ax3.set_title('Complementary CDF of Operational Time Before Detection', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    # Add percentile annotations
    for percentile in [25, 50, 75, 90]:
        value = np.percentile(operational_time_array, percentile)
        prob = 1 - percentile/100
        ax3.axhline(prob, color='gray', linestyle=':', alpha=0.5)
        ax3.axvline(value, color='gray', linestyle=':', alpha=0.5)
        ax3.text(value, prob + 0.02, f'p{percentile}', fontsize=8)

    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_filename}")
    plt.close()

    return {
        'total_compute_before_detection': total_compute_array,
        'detection_years': np.array(detection_year_samples),
        'operational_time_before_detection': operational_time_array,
        'percentiles': {
            'p10': np.percentile(total_compute_array, 10),
            'p25': np.percentile(total_compute_array, 25),
            'p50': np.percentile(total_compute_array, 50),
            'p75': np.percentile(total_compute_array, 75),
            'p90': np.percentile(total_compute_array, 90),
        },
        'mean': np.mean(total_compute_array),
        'std': np.std(total_compute_array),
        'output_file': output_filename
    }

def dark_compute_vs_workers(agreement_start_year=2028, num_samples=1000,
                                output_filename='dark_compute_vs_workers.png',
                                threshold=500000):
    """
    Calculate and plot expected (mean) H100 equivalents produced before detection vs number of workers.

    This function evaluates the expected total compute produced across different workforce sizes,
    helping to understand the trade-off between detection risk (more workers = faster detection)
    and production capacity (more workers = higher production rate).

    Args:
        agreement_start_year: The year the covert agreement/construction begins (default: 2028)
        num_samples: Number of Monte Carlo samples per worker count (default: 1000)
        output_filename: Filename for the output plot (default: 'expected_compute_vs_workers.png')
        threshold: Threshold for computing P(compute > threshold) (default: 500000)

    Returns:
        dict: Contains:
            - num_workers_array: Array of worker counts evaluated
            - expected_compute: Array of expected (mean) H100 equivalents produced
            - prob_exceeds_threshold: Array of probabilities that compute exceeds threshold
            - threshold: The threshold value used (default: 500000)
            - output_file: Path to saved visualization
    """
    # Generate log-spaced worker counts from 50 to 10,000
    num_workers_points = np.logspace(np.log10(50), np.log10(10000), 10).astype(int)

    mean_values = []
    prob_exceeds_values = []

    print(f"Calculating expected compute for {len(num_workers_points)} worker counts (using {num_samples} samples each)...")

    for i, num_workers in enumerate(num_workers_points):
        print(f"  {i+1}/{len(num_workers_points)}: {num_workers:,} workers...", end='')

        # Run compute_produced_before_detection for this worker count
        results = compute_produced_before_detection(
            agreement_start_year=agreement_start_year,
            num_workers=num_workers,
            num_samples=num_samples,
            max_years=15,
            output_filename=None  # Don't save individual plots
        )

        mean_values.append(results['mean'])

        # Calculate probability of exceeding threshold
        prob_exceeds = np.mean(results['total_compute_before_detection'] > threshold)
        prob_exceeds_values.append(prob_exceeds)

        print(f" Mean: {results['mean']/1e3:.1f}K, P(>{threshold/1e3:.0f}K): {prob_exceeds:.2%}")

    # Convert to arrays
    mean_array = np.array(mean_values)
    prob_exceeds_array = np.array(prob_exceeds_values)

    # Create visualization
    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot probability of exceeding threshold
    ax.plot(num_workers_points, prob_exceeds_array, 'k-', linewidth=2.5, marker='o')

    ax.set_xlabel('Number of Workers', fontsize=11)
    ax.set_ylabel(f'P(Compute produced before detection\n> {threshold/1e3:.0f}K H100 equivalents)', fontsize=11)
    ax.set_xscale('log')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, which='both')

    plt.title(f'Probability of Producing >{threshold/1e3:.0f}K H100 Equivalents Before Detection\n(Agreement starts in {agreement_start_year})',
              fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_filename}")

    return {
        'num_workers_array': num_workers_points,
        'expected_compute': mean_array,
        'prob_exceeds_threshold': prob_exceeds_array,
        'threshold': threshold,
        'output_file': output_filename
    }

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":

    agreement_start_year = 2028

    # Example 2: Compute produced before detection
    print("\n" + "=" * 70)
    print("EXAMPLE 2: COMPUTE PRODUCED BEFORE DETECTION")
    print("=" * 70)
    print()
    print(f"Calculating total compute produced before detection...")
    print(f"  Agreement start year: {agreement_start_year}")
    print(f"  Number of workers: 200")
    print()

    results_before_detection = compute_produced_before_detection(
        agreement_start_year=agreement_start_year,
        num_workers=100,
        num_samples=10000,
        output_filename='compute_before_detection.png'
    )

    print()
    print("RESULTS:")
    print(f"  Mean total compute: {results_before_detection['mean']/1e3:.1f}K H100 equivalents")
    print(f"  Std deviation: {results_before_detection['std']/1e3:.1f}K H100 equivalents")
    print()
    print("  Percentiles (H100 equivalents):")
    print(f"    10th: {results_before_detection['percentiles']['p10']/1e3:.1f}K")
    print(f"    25th: {results_before_detection['percentiles']['p25']/1e3:.1f}K")
    print(f"    50th (median): {results_before_detection['percentiles']['p50']/1e3:.1f}K")
    print(f"    75th: {results_before_detection['percentiles']['p75']/1e3:.1f}K")
    print(f"    90th: {results_before_detection['percentiles']['p90']/1e3:.1f}K")
    print()
    print("=" * 70)
