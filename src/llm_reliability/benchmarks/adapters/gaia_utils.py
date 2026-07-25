"""
Purpose
-------
Provide utility functions specifically designed for processing GAIA outputs.

Responsibilities
----------------
- Strip punctuation and standardize case for evaluation alignment
"""

from __future__ import annotations

import string


def normalize_gaia_answer(answer: str) -> str:
    """
    Normalize a GAIA answer for evaluation.
    Converts to lowercase and strips trailing/leading punctuation to increase
    resilience against minor formatting discrepancies.
    """
    if not isinstance(answer, str):
        return ""

    # Lowercase
    ans = answer.lower().strip()

    # Remove trailing/leading punctuation
    ans = ans.strip(string.punctuation)

    return ans
