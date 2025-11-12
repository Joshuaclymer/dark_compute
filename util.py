import numpy as np


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
