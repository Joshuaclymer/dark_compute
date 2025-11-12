import numpy as np
from scipy.stats import gamma
import pymetalog as pm

# Cache for metalog distributions to avoid re-fitting the same distribution
_metalog_cache = {}

def clear_metalog_cache():
    """Clear the metalog distribution cache. Call this when parameters change."""
    global _metalog_cache
    _metalog_cache = {}


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
    if cache_key not in _metalog_cache:
        # Fit metalog to the three quantiles
        # The metalog expects x values (quantiles) and their corresponding probabilities
        x = [p25, p50, p75]
        probs = [0.25, 0.5, 0.75]

        # Fit a 3-term bounded metalog with bounds [0, 1]
        m = pm.metalog(x=x, probs=probs, bounds=[0, 1], boundedness='b', term_limit=3, term_lower_bound=2)
        _metalog_cache[cache_key] = m
    else:
        m = _metalog_cache[cache_key]

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
    if cache_key not in _metalog_cache:
        # Fit metalog to the three quantiles
        # The metalog expects x values (quantiles) and their corresponding probabilities
        x = [p25, p50, p75]
        probs = [0.25, 0.5, 0.75]

        # Fit a 3-term semi-bounded metalog with lower bound at 0
        m = pm.metalog(x=x, probs=probs, bounds=[0], boundedness='sl', term_limit=3, term_lower_bound=2)
        _metalog_cache[cache_key] = m
    else:
        m = _metalog_cache[cache_key]

    # Sample from the distribution using pymetalog's rmetalog function
    # rmetalog returns an array, so we take the first element and convert to float
    result = float(pm.rmetalog(m, n=1, term=3)[0])

    # Ensure result is non-negative
    result = max(result, 0)

    return result
