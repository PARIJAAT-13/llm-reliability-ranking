"""
Confidence intervals module.

Computes bootstrap confidence intervals for a given sequence of numbers.
"""

from collections.abc import Sequence

import numpy as np

from llm_reliability.statistics.result_models import ConfidenceIntervalResult


def compute_bootstrap_ci(
    data: Sequence[float],
    confidence_level: float = 0.95,
    n_resamples: int = 2000,
    seed: int | None = 42,
) -> ConfidenceIntervalResult:
    """Compute bootstrap confidence intervals for the mean of the data.

    Parameters
    ----------
    data : Sequence[float]
        The input sequence of numeric data.
    confidence_level : float, default 0.95
        The confidence level, strictly between 0.0 and 1.0.
    n_resamples : int, default 2000
        Number of bootstrap resamples.
    seed : int, optional
        Random seed for deterministic bootstrapping.

    Returns
    -------
    ConfidenceIntervalResult
        Pydantic model containing the lower and upper bounds.
    """
    if len(data) == 0:
        raise ValueError("Cannot compute bootstrap confidence intervals on empty data.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be strictly between 0.0 and 1.0.")

    arr = np.asarray(data, dtype=float)
    rng = np.random.default_rng(seed)

    # Generate bootstrap samples of the mean
    boot_means = []
    for _ in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))

    alpha = 1.0 - confidence_level
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    lower = float(np.percentile(boot_means, lower_pct))
    upper = float(np.percentile(boot_means, upper_pct))

    return ConfidenceIntervalResult(
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
    )
