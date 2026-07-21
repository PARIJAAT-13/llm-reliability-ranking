"""
Utilities for the Statistical Analysis Engine.

Handles input validations and helper functions for summary statistics.
"""

import numpy as np
import pandas as pd
from typing import Sequence

from llm_reliability.records.ranking import RankingRecord
from llm_reliability.statistics.result_models import SummaryStatistics


def validate_rankings(ranking1: RankingRecord, ranking2: RankingRecord) -> None:
    """Validate two RankingRecords for compatibility.

    Raises ValueError if:
    - Either ranking is empty.
    - Mismatched ranking lengths.
    - Duplicate agent IDs within a ranking.
    - Any missing/None scores.
    """
    if not ranking1.rankings or not ranking2.rankings:
        raise ValueError("Rankings list cannot be empty.")

    if len(ranking1.rankings) != len(ranking2.rankings):
        raise ValueError(
            f"Mismatched ranking lengths: {len(ranking1.rankings)} vs {len(ranking2.rankings)}."
        )

    # Check duplicates and missing scores in ranking1
    agents1 = []
    for agent, score in ranking1.rankings:
        if score is None:
            raise ValueError(f"Agent '{agent}' in first ranking has a missing/None score.")
        agents1.append(agent)
    if len(agents1) != len(set(agents1)):
        raise ValueError("Duplicate agent IDs found in first ranking.")

    # Check duplicates and missing scores in ranking2
    agents2 = []
    for agent, score in ranking2.rankings:
        if score is None:
            raise ValueError(f"Agent '{agent}' in second ranking has a missing/None score.")
        agents2.append(agent)
    if len(agents2) != len(set(agents2)):
        raise ValueError("Duplicate agent IDs found in second ranking.")

    # Validate that they cover the exact same set of agents
    if set(agents1) != set(agents2):
        raise ValueError("The two rankings must contain the exact same set of agents.")


def calculate_summary_statistics(data: Sequence[float]) -> SummaryStatistics:
    """Calculate summary statistics for a sequence of numeric data.

    Parameters
    ----------
    data : Sequence[float]
        The input data.

    Returns
    -------
    SummaryStatistics
        Pydantic model containing summary stats.
    """
    if len(data) == 0:
        raise ValueError("Cannot calculate summary statistics on empty data.")

    arr = np.asarray(data, dtype=float)
    count = len(arr)
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    # Unbiased sample variance/std_dev (ddof=1) if count > 1, else 0.0
    variance = float(np.var(arr, ddof=1)) if count > 1 else 0.0
    std_dev = float(np.std(arr, ddof=1)) if count > 1 else 0.0
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))

    return SummaryStatistics(
        mean=mean,
        median=median,
        variance=variance,
        std_dev=std_dev,
        min_val=min_val,
        max_val=max_val,
        q1=q1,
        q3=q3,
        count=count,
    )
