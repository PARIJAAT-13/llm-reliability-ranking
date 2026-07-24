"""
Fault tolerance metric computation.

Formula
-------
Fault tolerance measures an agent's ability to produce correct outputs
even when a failure has been deliberately injected into the execution
environment (e.g. a missing tool, a network timeout, a corrupted context).

    normal_sr       = mean score of non-fault-injected evaluations
    fault_sr        = mean score of fault-injected evaluations
    fault_tolerance = fault_sr / normal_sr   if normal_sr > 0
                    = 0.0                    if normal_sr == 0 and fault_sr == 0
                    = 1.0                    if normal_sr == 0 and fault_sr > 0
                    clamped to [0, 1]

Intuition: a fault tolerance of 1.0 means the agent is equally effective
whether or not faults are injected.  A value close to 0 means the agent
collapses entirely in the presence of injected failures.

Normal records have ``fault_injected == False``.
Fault-injected records have ``fault_injected == True``.

Raises ValueError if no fault-injected evaluations are present.
"""

import numpy as np

from llm_reliability.records.evaluation import EvaluationRecord


def compute_fault_tolerance(evaluations: list[EvaluationRecord]) -> float:
    """Compute fault tolerance from a list of EvaluationRecords.

    Args:
        evaluations: Non-empty list of EvaluationRecord instances. Must
            contain at least one fault-injected record.

    Returns:
        A float in [0, 1].

    Raises:
        ValueError: If *evaluations* is empty or contains no fault-injected records.
    """
    if not evaluations:
        raise ValueError("Cannot compute fault_tolerance from empty evaluations.")

    normal = [ev for ev in evaluations if not ev.fault_injected]
    faulted = [ev for ev in evaluations if ev.fault_injected]

    if not faulted:
        raise ValueError("No fault-injected evaluations found. Cannot compute fault_tolerance.")

    normal_sr = float(np.mean([ev.score for ev in normal])) if normal else 0.0
    fault_sr = float(np.mean([ev.score for ev in faulted]))

    if normal_sr == 0.0:
        fault_tolerance = 0.0 if fault_sr == 0.0 else 1.0
    else:
        fault_tolerance = float(np.clip(fault_sr / normal_sr, 0.0, 1.0))

    return fault_tolerance
