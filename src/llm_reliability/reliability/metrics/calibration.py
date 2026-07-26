from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field

from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.utils.serialization import SerializableModel


class CalibrationResult(SerializableModel):
    ece: float = Field(ge=0.0, le=1.0)
    mce: float = Field(ge=0.0, le=1.0)
    brier_score: float = Field(ge=0.0, le=1.0)
    n_bins: int = Field(ge=1)
    bin_accuracies: list[float]
    bin_confidences: list[float]
    bin_counts: list[int]
    n_samples: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


def compute_ece(
    confidences: list[float],
    outcomes: list[bool],
    n_bins: int = 10,
) -> tuple[float, list[float], list[float], list[int]]:
    if not confidences or len(confidences) != len(outcomes):
        raise ValueError("confidences and outcomes must be non-empty and same length")

    conf = np.array(confidences, dtype=float)
    outc = np.array(outcomes, dtype=float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs: list[float] = []
    bin_confs: list[float] = []
    bin_counts: list[int] = []

    ece = 0.0
    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        in_bin = (conf > lo) & (conf <= hi)
        if i == 0:
            in_bin = (conf >= lo) & (conf <= hi)

        count = int(np.sum(in_bin))
        bin_counts.append(count)
        if count > 0:
            bin_acc = float(np.mean(outc[in_bin]))
            bin_conf = float(np.mean(conf[in_bin]))
            ece += count * abs(bin_acc - bin_conf)
        else:
            bin_acc = 0.0
            bin_conf = 0.0
        bin_accs.append(bin_acc)
        bin_confs.append(bin_conf)

    ece /= len(confidences)
    return ece, bin_accs, bin_confs, bin_counts


def compute_mce(
    confidences: list[float],
    outcomes: list[bool],
    n_bins: int = 10,
) -> float:
    ece, bin_accs, bin_confs, bin_counts = compute_ece(confidences, outcomes, n_bins)
    gaps = [abs(a - c) for a, c, cnt in zip(bin_accs, bin_confs, bin_counts) if cnt > 0]
    return max(gaps) if gaps else 0.0


def compute_brier_score(
    confidences: list[float],
    outcomes: list[bool],
) -> float:
    if not confidences or len(confidences) != len(outcomes):
        raise ValueError("confidences and outcomes must be non-empty and same length")
    conf = np.array(confidences, dtype=float)
    outc = np.array(outcomes, dtype=float)
    return float(np.mean((conf - outc) ** 2))


def compute_calibration(
    evaluations: list[EvaluationRecord],
    n_bins: int = 10,
) -> CalibrationResult:
    if not evaluations:
        raise ValueError("Cannot compute calibration from empty evaluations")

    confidences = [ev.score for ev in evaluations]
    outcomes = [ev.success for ev in evaluations]

    ece, bin_accs, bin_confs, bin_counts = compute_ece(confidences, outcomes, n_bins)
    mce_val = compute_mce(confidences, outcomes, n_bins)
    brier = compute_brier_score(confidences, outcomes)

    return CalibrationResult(
        ece=ece,
        mce=mce_val,
        brier_score=brier,
        n_bins=n_bins,
        bin_accuracies=bin_accs,
        bin_confidences=bin_confs,
        bin_counts=bin_counts,
        n_samples=len(evaluations),
    )


def _reliability_diagram_data(
    evaluations: list[EvaluationRecord],
    n_bins: int = 10,
) -> dict:
    result = compute_calibration(evaluations, n_bins)
    return {
        "bin_accuracies": result.bin_accuracies,
        "bin_confidences": result.bin_confidences,
        "bin_counts": result.bin_counts,
        "ece": result.ece,
        "mce": result.mce,
        "n_bins": n_bins,
    }
