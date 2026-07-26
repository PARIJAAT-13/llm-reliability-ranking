"""
Assumption checking module.

Validates statistical assumptions before running hypothesis tests.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import shapiro


def check_normality(differences: Sequence[float]) -> tuple[bool, str | None]:
    """Check if the differences are normally distributed using the Shapiro-Wilk test.

    Parameters
    ----------
    differences : Sequence[float]
        The paired differences.

    Returns
    -------
    tuple[bool, str | None]
        - bool: True if assumptions are met, False otherwise.
        - str | None: Warning message if assumptions are violated.
    """
    diff_arr = np.asarray(differences, dtype=float)
    if len(diff_arr) < 3:
        return (
            False,
            f"Sample size too small to validate normality (n={len(diff_arr)} < 3).",
        )
    if np.ptp(diff_arr) == 0:
        return True, None

    # Shapiro-Wilk test for normality
    stat, p_val = shapiro(diff_arr)
    if p_val < 0.05:
        return (
            False,
            f"Normality assumption violated (Shapiro-Wilk p-value = {p_val:.4f} < 0.05).",
        )

    return True, None
