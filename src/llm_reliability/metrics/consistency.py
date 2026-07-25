"""
Consistency metric computation.

Formula
-------
Repeated-run consistency measures the degree to which an agent produces
the same outcome when the same task is executed multiple times.

    consistency = 1 - (std of per-task success rates)

More precisely, for each unique task_id we compute its *success rate*
(fraction of runs that succeeded).  We then take the mean of those
per-task success rates and subtract the standard deviation.

    per_task_sr[t] = successes[t] / runs[t]   for each task t
    consistency    = mean(per_task_sr) - std(per_task_sr)
                   clamped to [0, 1]

Intuition: if every task succeeds 100 % of the time, std = 0 and
consistency = 1.  If some tasks always succeed and others always fail,
std is high and consistency drops.  An agent that is randomly inconsistent
across tasks will have a consistency close to 0.

Minimum two evaluations per task are required to compute variance.
If every task has exactly one evaluation the std component is 0 and
consistency equals the overall success rate.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from llm_reliability.records.evaluation import EvaluationRecord


def compute_consistency(evaluations: list[EvaluationRecord]) -> float:
    """Compute repeated-run consistency from a list of EvaluationRecords.

    Args:
        evaluations: Non-empty list of EvaluationRecord instances.

    Returns:
        A float in [0, 1] representing consistency.

    Raises:
        ValueError: If *evaluations* is empty.
    """
    if not evaluations:
        raise ValueError("Cannot compute consistency from empty evaluations.")

    # Group success flags by task_id
    task_successes: dict[str, list[bool]] = defaultdict(list)
    for ev in evaluations:
        task_successes[ev.task_id].append(ev.success)

    # Per-task success rate
    per_task_rates = np.array(
        [np.mean(flags) for flags in task_successes.values()],
        dtype=float,
    )

    mean_sr = float(np.mean(per_task_rates))
    std_sr = float(np.std(per_task_rates))

    consistency = float(np.clip(mean_sr - std_sr, 0.0, 1.0))
    return consistency
