import numpy as np
from scipy.stats import gamma
import pymetalog as pm
import numpy as np
from scipy.interpolate import interp1d
from scipy import stats
from typing import Dict

class Cache():
    """A simple cache class to store and retrieve computed values."""
    def __init__(self):
        self.metalog_distributions = {}
        self.composite_detection_distributions = {}

    def clear(self):
        """Clear the entire cache."""
        self.metalog_distributions = {}
        self.composite_detection_distributions = {}

_cache = Cache()

def sample_from_log_normal(median: float, relative_sigma: float) -> float:
    """
    Sample from a lognormal distribution given a median and relative standard deviation.

    Args:
        median: The median of the lognormal distribution
        relative_sigma: The relative standard deviation (sigma / mean) of the lognormal distribution

    Returns:
        A sample from the lognormal distribution
    """
    # Handle edge cases where median is zero or negative
    if median <= 0:
        return 0.0

    sigma_log = np.sqrt(np.log(1 + relative_sigma**2))
    mu_log = np.log(median)
    return np.random.lognormal(mean=mu_log, sigma=sigma_log)


def sample_from_metalog_3term(p25: float, p50: float, p75: float) -> float:
    """
    Sample from a bounded (0,1) distribution using a 3-term metalog.

    Uses the pymetalog library to fit a metalog distribution to the percentiles.

    Args:
        p25: 25th percentile value (between 0 and 1)
        p50: 50th percentile (median) value (between 0 and 1)
        p75: 75th percentile value (between 0 and 1)

    Returns:
        A sample from the distribution bounded to (0, 1)
    """
    # Clamp percentiles to valid range
    p25 = np.clip(p25, 1e-6, 1 - 1e-6)
    p50 = np.clip(p50, 1e-6, 1 - 1e-6)
    p75 = np.clip(p75, 1e-6, 1 - 1e-6)

    # Create cache key
    cache_key = ('bounded', p25, p50, p75)

    # Check if we've already fitted this distribution
    if cache_key not in _cache.metalog_distributions:
        # Fit metalog to the three quantiles
        # The metalog expects x values (quantiles) and their corresponding probabilities
        x = [p25, p50, p75]
        probs = [0.25, 0.5, 0.75]

        # Fit a 3-term bounded metalog with bounds [0, 1]
        m = pm.metalog(x=x, probs=probs, bounds=[0, 1], boundedness='b', term_limit=3, term_lower_bound=2)
        _cache.metalog_distributions[cache_key] = m
    else:
        m = _cache.metalog_distributions[cache_key]

    # Sample from the distribution using pymetalog's rmetalog function
    # rmetalog returns an array, so we take the first element and convert to float
    result = float(pm.rmetalog(m, n=1, term=3)[0])

    # Ensure result is in valid range (should already be bounded by metalog)
    result = np.clip(result, 0, 1)

    return result


def sample_from_metalog_3term_semi_bounded(p25: float, p50: float, p75: float) -> float:
    """
    Sample from a semi-bounded (lower bound of 0, no upper bound) distribution using a 3-term metalog.

    Uses the pymetalog library to fit a metalog distribution to the percentiles.
    This is suitable for multipliers where we want to allow arbitrarily large values.

    Args:
        p25: 25th percentile value (must be > 0)
        p50: 50th percentile (median) value (must be > 0)
        p75: 75th percentile value (must be > 0)

    Returns:
        A sample from the distribution bounded at 0 from below
    """
    # Ensure percentiles are positive
    p25 = max(p25, 1e-6)
    p50 = max(p50, 1e-6)
    p75 = max(p75, 1e-6)

    # Create cache key
    cache_key = ('semi_bounded', p25, p50, p75)

    # Check if we've already fitted this distribution
    if cache_key not in _cache.metalog_distributions:
        # Fit metalog to the three quantiles
        # The metalog expects x values (quantiles) and their corresponding probabilities
        x = [p25, p50, p75]
        probs = [0.25, 0.5, 0.75]

        # Fit a 3-term semi-bounded metalog with lower bound at 0
        m = pm.metalog(x=x, probs=probs, bounds=[0], boundedness='sl', term_limit=3, term_lower_bound=2)
        _cache.metalog_distributions[cache_key] = m
    else:
        m = _cache.metalog_distributions[cache_key]

    # Sample from the distribution using pymetalog's rmetalog function
    # rmetalog returns an array, so we take the first element and convert to float
    result = float(pm.rmetalog(m, n=1, term=3)[0])

    # Ensure result is non-negative
    result = max(result, 0)

    return result

def build_composite_detection_distribution(
    labor_by_year: Dict[float, int],
    mean_detection_time_100_workers: float,
    mean_detection_time_1000_workers: float,
    variance_theta: float
) -> tuple:
    """
    Build composite distribution for detection time accounting for variable labor.

    This is the expensive calculation that should be done once and reused across simulations.

    Args:
        labor_by_year: Dict mapping years to total number of workers at that time
        mean_detection_time_100_workers: Mean detection time for 100 workers
        mean_detection_time_1000_workers: Mean detection time for 1000 workers
        variance_theta: Variance parameter of detection time given num workers

    Returns:
        Tuple of (sorted_years, prob_ranges, A, B, variance_theta) where:
        - sorted_years: List of years in chronological order
        - prob_ranges: List of tuples (year_start, year_end, labor, k, theta, cum_start, cum_end)
        - A, B: Detection time constants
        - variance_theta: Passed through for convenience
    """
    if not labor_by_year:
        return ([], [], None, None, variance_theta)

    # Sort years chronologically
    sorted_years = sorted(labor_by_year.keys())

    # Compute detection time constants
    x1, mu1 = 100, mean_detection_time_100_workers
    x2, mu2 = 1000, mean_detection_time_1000_workers
    B = np.log(mu1 / mu2) / np.log(np.log10(x2) / np.log10(x1))
    A = mu1 * (np.log10(x1) ** B)

    # Build composite distribution by calculating cumulative probabilities
    cumulative_prob = 0.0
    prob_ranges = []

    for i, year in enumerate(sorted_years):
        labor = labor_by_year[year]

        if labor <= 0:
            continue  # Skip periods with no workers

        # Calculate gamma parameters for this labor level
        mu = A / (np.log10(labor) ** B)
        k = mu / variance_theta

        # Determine the time range for this period
        year_start = year
        year_end = sorted_years[i + 1] if i + 1 < len(sorted_years) else year + 100

        # Calculate probability mass in this interval using CDF differences
        cdf_start = stats.gamma.cdf(year_start, a=k, scale=variance_theta)
        cdf_end = stats.gamma.cdf(year_end, a=k, scale=variance_theta)
        prob_mass = cdf_end - cdf_start

        cum_prob_end = cumulative_prob + prob_mass
        prob_ranges.append((year_start, year_end, labor, k, variance_theta, cumulative_prob, cum_prob_end))
        cumulative_prob = cum_prob_end

    return (sorted_years, prob_ranges, A, B, variance_theta)


def sample_detection_time_from_composite(prob_ranges) -> float:
    """
    Sample a detection time from the precomputed composite distribution.

    Args:
        prob_ranges: Precomputed probability ranges from build_composite_detection_distribution

    Returns:
        float: Sampled detection time in years
    """
    if not prob_ranges:
        return float('inf')

    u = np.random.uniform(0, 1)

    # Find which range the sample falls into
    time_of_detection = float('inf')
    for year_start, year_end, _, k, theta, cum_start, cum_end in prob_ranges:
        if cum_start <= u < cum_end:
            # Map u within this range back to the gamma distribution
            if cum_end > cum_start:
                u_normalized = (u - cum_start) / (cum_end - cum_start)
                cdf_start = stats.gamma.cdf(year_start, a=k, scale=theta)
                cdf_target = cdf_start + u_normalized * (stats.gamma.cdf(year_end, a=k, scale=theta) - cdf_start)
                time_of_detection = stats.gamma.ppf(cdf_target, a=k, scale=theta)
            else:
                time_of_detection = year_start
            break

    return time_of_detection


def lr_over_time_vs_num_workers(
    labor_by_year: Dict[float, int],
    mean_detection_time_100_workers: float,
    mean_detection_time_1000_workers: float,
    variance_theta: float
) -> Dict[float, float]:
    """
    Calculate likelihood ratios over time accounting for variable labor.

    This function:
    1. Checks cache for existing composite distribution (or builds and caches it)
    2. Samples detection time from the composite distribution
    3. Calculates LRs for each year

    Args:
        labor_by_year: Dict mapping years to total number of workers at that time
        mean_detection_time_100_workers: Mean detection time for 100 workers
        mean_detection_time_1000_workers: Mean detection time for 1000 workers
        variance_theta: Variance parameter of detection time given num workers

    Returns:
        Dict[float, float]: Mapping of years to likelihood ratios
    """
    if not labor_by_year:
        return {}

    # Create cache key from the input parameters
    # Convert labor_by_year dict to a hashable tuple for caching
    labor_tuple = tuple(sorted(labor_by_year.items()))
    cache_key = (labor_tuple, mean_detection_time_100_workers, mean_detection_time_1000_workers, variance_theta)

    # Check if composite distribution is already cached
    if cache_key not in _cache.composite_detection_distributions:
        # Build and cache the composite distribution
        sorted_years, prob_ranges, A, B, variance_theta_out = build_composite_detection_distribution(
            labor_by_year=labor_by_year,
            mean_detection_time_100_workers=mean_detection_time_100_workers,
            mean_detection_time_1000_workers=mean_detection_time_1000_workers,
            variance_theta=variance_theta
        )
        _cache.composite_detection_distributions[cache_key] = {
            'sorted_years': sorted_years,
            'prob_ranges': prob_ranges,
            'A': A,
            'B': B,
            'variance_theta': variance_theta_out
        }
    else:
        # Retrieve cached distribution
        cached = _cache.composite_detection_distributions[cache_key]
        sorted_years = cached['sorted_years']
        prob_ranges = cached['prob_ranges']
        A = cached['A']
        B = cached['B']
        variance_theta = cached['variance_theta']

    # Sample detection time from the composite distribution
    time_of_detection = sample_detection_time_from_composite(prob_ranges)

    # Calculate likelihood ratios for each year
    lr_by_year = {}
    for year in sorted_years:
        if year >= time_of_detection:
            # Detection has occurred by this year
            lr_by_year[year] = 100.0  # Very high LR - project detected
        else:
            labor = labor_by_year[year]
            if labor <= 0:
                lr_by_year[year] = 1.0
                continue

            # Calculate survival probability using gamma distribution for current labor level
            mu = A / (np.log10(labor) ** B)
            k = mu / variance_theta

            p_not_detected_given_fab = stats.gamma.sf(year, a=k, scale=variance_theta)

            # LR = P(evidence | fab exists) / P(evidence | no fab)
            lr = p_not_detected_given_fab / 1.0

            lr_by_year[year] = max(lr, 0.001)  # Floor at 0.001

    return lr_by_year