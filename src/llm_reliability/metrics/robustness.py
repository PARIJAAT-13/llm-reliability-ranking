"""
Perturbation robustness metric computation.

Formula
-------
Robustness measures how well an agent maintains performance when the input
prompt is slightly perturbed (e.g. rephrased, injected with noise, or
translated).

    baseline_sr   = mean score of non-perturbed evaluations
    perturbed_sr  = mean score of perturbed evaluations
    robustness    = perturbed_sr / baseline_sr   if baseline_sr > 0
                  = 0.0                          if baseline_sr == 0 and perturbed_sr == 0
                  = 1.0                          if baseline_sr == 0 and perturbed_sr > 0
                  clamped to [0, 1]

Intuition: a robustness of 1.0 means perturbed performance equals baseline.
Values < 1.0 indicate degradation under perturbation.  Values are clamped so
that a perturbed score higher than baseline does not exceed 1.0.

Baseline records have ``perturbation is None``.
Perturbed records have a non-None ``perturbation`` field.

Raises ValueError if no perturbed evaluations are found — calling code should
check before calling when perturbations are optional.
"""

from __future__ import annotations

import numpy as np

from llm_reliability.records.evaluation import EvaluationRecord


def compute_robustness(evaluations: list[EvaluationRecord]) -> float:
    """Compute perturbation robustness from a list of EvaluationRecords.

    Args:
        evaluations: Non-empty list of EvaluationRecord instances.  Must
            contain at least one perturbed record.

    Returns:
        A float in [0, 1].

    Raises:
        ValueError: If *evaluations* is empty or contains no perturbed records.
    """
    if not evaluations:
        raise ValueError("Cannot compute robustness from empty evaluations.")

    baseline = [ev for ev in evaluations if ev.perturbation is None]
    perturbed = [ev for ev in evaluations if ev.perturbation is not None]

    if not perturbed:
        raise ValueError("No perturbed evaluations found. Cannot compute robustness.")

    baseline_sr = float(np.mean([ev.score for ev in baseline])) if baseline else 0.0
    perturbed_sr = float(np.mean([ev.score for ev in perturbed]))

    if baseline_sr == 0.0:
        robustness = 0.0 if perturbed_sr == 0.0 else 1.0
    else:
        robustness = float(np.clip(perturbed_sr / baseline_sr, 0.0, 1.0))

    return robustness
