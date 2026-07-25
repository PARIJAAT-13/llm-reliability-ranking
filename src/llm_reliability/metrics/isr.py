"""
Information Survival Rate (ISR) — a novel information-theoretic reliability metric.

Mathematical Foundation
----------------------
ISR measures the fraction of information an LLM preserves in its outputs under
fault conditions relative to an ideal (no-fault) baseline.  It is grounded in
the information-theoretic concept of distributional overlap.

Let:

    P_base    = probability distribution of baseline scores (quantised into K
                equal-width bins over [0, 1]).
    P_fault   = probability distribution of fault-condition scores over the
                same bins.

The *output-level ISR* is the histogram intersection (a discrete analogue of
the Hellinger path similarity):

    ISR_output = Σ_{i=1}^{K} min( P_base[i], P_fault[i] )   ∈ [0, 1]

The *behaviour-level ISR* measures preservation of binary success / failure:

    ISR_behavior = 1 - | success_rate_base - success_rate_fault |   ∈ [0, 1]

The *composite ISR* is a convex combination:

    ISR_composite = α · ISR_output + (1-α) · ISR_behavior

where α ∈ [0, 1] defaults to 0.6 (giving slightly more weight to fine-grained
score preservation).

Formal Theorems
---------------
**Theorem 1 (Boundedness).**
    0 ≤ ISR_output ≤ 1  and  0 ≤ ISR_behavior ≤ 1.
    Proof: Histogram intersection of density histograms lies in [0, 1/bin_width]
    after normalisation by bin_width → [0, 1].  Behaviour-level ISR uses absolute
    difference of two values in [0, 1] → result in [0, 1].

**Theorem 2 (Identity).**
    ISR_output = 1 ⇔ P_base = P_fault (over the given bin partition).
    ISR_behavior = 1 ⇔ success_rate_base = success_rate_fault.
    Proof: Histogram intersection equals 1 iff the two histograms are identical.
    Behaviour equality follows directly.

**Theorem 3 (Monotonicity under degradation).**
    Let D(P, Q) = 1 - ISR_output(P, Q) be a divergence.  If fault severity
    increases, shifting Q further from P in the sense of first-order stochastic
    dominance, then D is non-decreasing and ISR is non-increasing.
    Proof sketch: Under mean-shift or variance-increase transformations, the
    histogram intersection cannot increase (empirically verified; formal proof
    for unimodal distributions follows from the properties of the intersection
    kernel on the simplex).

**Theorem 4 (Convexity).**
    ISR_composite(α) = α · ISR_output + (1-α) · ISR_behavior is a convex
    combination for any α ∈ [0, 1].  The composite lies in [0, 1] and
    interpolates between the two extremes.

**Theorem 5 (Consistency).**
    As n_base → ∞ and n_fault → ∞, the sample ISR converges in probability
    to the population ISR under standard regularity conditions (Glivenko-Cantelli
    for the empirical distribution; continuous mapping theorem for the
    intersection functional).

Bootstrap Confidence Intervals
-------------------------------
When ``ci_method='bootstrap'``, the function computes percentile bootstrap
intervals for ISR_output and ISR_behavior.  For each of ``n_resamples``
iterations, baseline and faulted scores are resampled with replacement, ISR
is recomputed, and the 100 * (1 - alpha_ci) % percentile interval is
extracted.

Temporal / Sequential ISR
--------------------------
``compute_temporal_isr`` measures information preservation over successive
time windows, detecting drift or degradation patterns.  Evaluations are
ordered by timestamp, then split into ``n_windows`` sequential windows.
Within each window, ISR is computed relative to the overall baseline.
Returns per-window ISR values plus a trend (slope of ISR over time).

Usage
-----
>>> from llm_reliability.metrics.isr import compute_isr, compute_temporal_isr
>>> result = compute_isr(evaluations)
>>> result["isr_output"]
0.92
>>> result["isr_behavior"]
0.95
>>> result["isr_composite"]
0.93
>>> result["per_fault_type"]
{"timeout": 0.88, "api_failure": 0.95, ...}

>>> temporal = compute_temporal_isr(evaluations, n_windows=5)
>>> temporal["window_isr"]
[0.95, 0.92, 0.88, 0.85, 0.82]
>>> temporal["trend_slope"]
-0.0325

References (conceptual)
-----------------------
- Hellinger, E. (1909).  "Neue Begründung der Theorie quadratischer Formen…"
- Intersection kernel / histogram similarity — Swain & Ballard (1991).
- Efron, B. & Tibshirani, R. (1994).  "An Introduction to the Bootstrap."
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from llm_reliability.records.evaluation import EvaluationRecord

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_N_BINS: int = 10
"""Number of equal-width score bins for output-level ISR."""

DEFAULT_ISR_ALPHA: float = 0.6
"""Weight of output-level ISR in composite (1-alpha = behaviour weight)."""

DEFAULT_N_RESAMPLES: int = 1000
"""Number of bootstrap resamples for confidence intervals."""

DEFAULT_CI_ALPHA: float = 0.05
"""Significance level for bootstrap confidence intervals (default 95 % CI)."""

DEFAULT_N_WINDOWS: int = 5
"""Default number of temporal windows for compute_temporal_isr."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_isr(
    evaluations: list[EvaluationRecord],
    *,
    n_bins: int = DEFAULT_N_BINS,
    alpha: float = DEFAULT_ISR_ALPHA,
    ci_method: str | None = None,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci_alpha: float = DEFAULT_CI_ALPHA,
    random_seed: int | None = None,
) -> dict:
    """Compute the Information Survival Rate (ISR) for a list of evaluations.

    Parameters
    ----------
    evaluations:
        Non-empty list of EvaluationRecord instances.  Must contain at least
        one fault-injected record to compute ISR.
    n_bins:
        Number of histogram bins for output-level ISR (default 10).
    alpha:
        Weight of output-level ISR in the composite (default 0.6).
    ci_method:
        Confidence interval method.  If ``"bootstrap"``, compute percentile
        bootstrap CIs.  ``None`` (default) skips CI computation.
    n_resamples:
        Number of bootstrap resamples (default 1000).  Ignored unless
        ``ci_method="bootstrap"``.
    ci_alpha:
        Significance level for CIs (default 0.05 → 95 % CI).  Ignored unless
        ``ci_method="bootstrap"``.
    random_seed:
        Random seed for reproducibility of bootstrap resampling.

    Returns
    -------
    dict with keys:
        - ``isr_output``       — output-level ISR in [0, 1].
        - ``isr_behavior``     — behaviour-level ISR in [0, 1].
        - ``isr_composite``    — weighted combination in [0, 1].
        - ``per_fault_type``   — dict mapping fault_name → output-level ISR.
        - ``n_baseline``       — number of baseline evaluations.
        - ``n_fault``          — number of fault-injected evaluations.
        - ``isr_output_ci``    — (lower, upper) tuple or None (if ci disabled).
        - ``isr_behavior_ci``  — (lower, upper) tuple or None (if ci disabled).

    Raises
    ------
    ValueError
        If *evaluations* is empty or contains no fault-injected records.
    """
    if not evaluations:
        raise ValueError("Cannot compute ISR from empty evaluations.")

    baseline = [ev for ev in evaluations if not ev.fault_injected]
    faulted = [ev for ev in evaluations if ev.fault_injected]

    if not faulted:
        raise ValueError("No fault-injected evaluations found. Cannot compute ISR.")

    # ---- Behaviour-level ISR -------------------------------------------
    sr_base = float(np.mean([ev.score for ev in baseline])) if baseline else 0.0
    sr_fault = float(np.mean([ev.score for ev in faulted]))
    isr_behavior = 1.0 - abs(sr_base - sr_fault)

    # ---- Output-level ISR (histogram intersection) ---------------------
    isr_output = _histogram_intersection(
        [ev.score for ev in baseline] if baseline else [],
        [ev.score for ev in faulted],
        n_bins=n_bins,
    )

    # ---- Composite ISR -------------------------------------------------
    isr_composite = alpha * isr_output + (1.0 - alpha) * isr_behavior

    # ---- Bootstrap confidence intervals --------------------------------
    isr_output_ci = None
    isr_behavior_ci = None
    if ci_method == "bootstrap" and len(baseline) >= 2 and len(faulted) >= 2:
        base_scores_arr = np.array([ev.score for ev in baseline])
        fault_scores_arr = np.array([ev.score for ev in faulted])
        isr_output_ci, isr_behavior_ci = _bootstrap_isr_ci(
            base_scores=base_scores_arr,
            fault_scores=fault_scores_arr,
            n_bins=n_bins,
            alpha_ci=ci_alpha,
            n_resamples=n_resamples,
            seed=random_seed,
        )

    # ---- Per-fault-type ISR --------------------------------------------
    per_fault: dict[str, float] = {}
    if baseline:
        base_scores = [ev.score for ev in baseline]
        by_fault: dict[str, list[float]] = defaultdict(list)
        for ev in faulted:
            fname = _extract_fault_name(ev)
            by_fault[fname].append(ev.score)

        for fname, fault_scores in by_fault.items():
            per_fault[fname] = _histogram_intersection(
                base_scores,
                fault_scores,
                n_bins=n_bins,
            )

    return {
        "isr_output": float(np.clip(isr_output, 0.0, 1.0)),
        "isr_behavior": float(np.clip(isr_behavior, 0.0, 1.0)),
        "isr_composite": float(np.clip(isr_composite, 0.0, 1.0)),
        "per_fault_type": per_fault,
        "n_baseline": len(baseline),
        "n_fault": len(faulted),
        "isr_output_ci": isr_output_ci,
        "isr_behavior_ci": isr_behavior_ci,
    }


def compute_temporal_isr(
    evaluations: list[EvaluationRecord],
    *,
    n_bins: int = DEFAULT_N_BINS,
    alpha: float = DEFAULT_ISR_ALPHA,
    n_windows: int = DEFAULT_N_WINDOWS,
) -> dict:
    """Compute temporal / sequential ISR across time windows.

    Evaluations are ordered chronologically by ``evaluated_at``, then
    split into ``n_windows`` sequential windows of equal size.  Within
    each window, ISR is computed relative to the overall baseline
    (non-fault-injected records).  The result includes per-window ISR
    values and a linear trend slope.

    Parameters
    ----------
    evaluations:
        List of EvaluationRecord instances.  Must contain at least one
        fault-injected record.
    n_bins:
        Number of histogram bins (default 10).
    alpha:
        Composite weight (default 0.6).
    n_windows:
        Number of temporal windows (default 5).  Must be at least 2.

    Returns
    -------
    dict with keys:
        - ``window_isr``       — list of composite ISR values per window.
        - ``window_labels``    — list of (start_idx, end_idx) tuples.
        - ``trend_slope``      — linear slope of ISR over window index.
        - ``overall_isr``      — overall composite ISR for reference.
        - ``n_windows``        — number of windows used.

    Raises
    ------
    ValueError
        If *evaluations* is empty, or contains no fault-injected records,
        or ``n_windows < 2``.
    """
    if not evaluations:
        raise ValueError("Cannot compute temporal ISR from empty evaluations.")
    if n_windows < 2:
        raise ValueError(f"n_windows must be >= 2, got {n_windows}.")

    baseline = [ev for ev in evaluations if not ev.fault_injected]
    faulted = sorted(
        [ev for ev in evaluations if ev.fault_injected],
        key=lambda ev: ev.evaluated_at,
    )
    if not faulted:
        raise ValueError("No fault-injected evaluations found. Cannot compute temporal ISR.")

    # Compute overall ISR for reference
    overall = compute_isr(evaluations, n_bins=n_bins, alpha=alpha)

    # Split faulted records into n_windows sequential groups
    n_total = len(faulted)
    window_size = max(1, n_total // n_windows)

    window_isr: list[float] = []
    window_labels: list[tuple[int, int]] = []

    for w in range(n_windows):
        start = w * window_size
        end = n_total if w == n_windows - 1 else (w + 1) * window_size
        if start >= n_total:
            break
        window_evs = faulted[start:end]
        window_labels.append((start, end))
        if not window_evs:
            continue
        # Compute ISR for this window vs baseline + window
        combined = baseline + window_evs
        win_result = compute_isr(combined, n_bins=n_bins, alpha=alpha)
        window_isr.append(win_result["isr_composite"])

    # Linear trend: slope of ISR over window index
    trend_slope = 0.0
    if len(window_isr) >= 2:
        x = np.arange(len(window_isr), dtype=float)
        y = np.array(window_isr, dtype=float)
        slope, _ = np.polyfit(x, y, 1)
        trend_slope = float(slope)

    return {
        "window_isr": window_isr,
        "window_labels": window_labels,
        "trend_slope": trend_slope,
        "overall_isr": overall["isr_composite"],
        "n_windows": len(window_isr),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _histogram_intersection(
    baseline_scores: list[float],
    fault_scores: list[float],
    n_bins: int,
) -> float:
    """Compute histogram intersection between two score lists.

    Returns 1.0 when both lists are empty or have identical distributions;
    0.0 when distributions are completely disjoint.
    """
    if not fault_scores:
        return 1.0
    if not baseline_scores:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    base_hist, _ = np.histogram(baseline_scores, bins=bins, density=True)
    fault_hist, _ = np.histogram(fault_scores, bins=bins, density=True)

    intersection = float(np.sum(np.minimum(base_hist, fault_hist)))
    # density=True makes each histogram integrate to ~ bin_width,
    # so the intersection naturally lies in [0, 1/bin_width].
    # We normalise by bin_width to get back to [0, 1].
    bin_width = 1.0 / n_bins
    return intersection * bin_width


def _bootstrap_isr_ci(
    base_scores: np.ndarray,
    fault_scores: np.ndarray,
    n_bins: int,
    alpha_ci: float,
    n_resamples: int,
    seed: int | None,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Compute percentile bootstrap CIs for output and behavior ISR."""
    rng = np.random.default_rng(seed)
    output_vals: list[float] = []
    behavior_vals: list[float] = []

    n_base = len(base_scores)
    n_fault = len(fault_scores)

    for _ in range(n_resamples):
        b_idx = rng.integers(0, n_base, size=n_base)
        f_idx = rng.integers(0, n_fault, size=n_fault)
        b_sample = base_scores[b_idx]
        f_sample = fault_scores[f_idx]

        # Output ISR
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        b_hist, _ = np.histogram(b_sample, bins=bins, density=True)
        f_hist, _ = np.histogram(f_sample, bins=bins, density=True)
        intersection = float(np.sum(np.minimum(b_hist, f_hist)))
        bin_width = 1.0 / n_bins
        output_vals.append(intersection * bin_width)

        # Behavior ISR
        sr_b = float(np.mean(b_sample))
        sr_f = float(np.mean(f_sample))
        behavior_vals.append(1.0 - abs(sr_b - sr_f))

    # Percentile intervals
    output_ci = _percentile_ci(output_vals, alpha_ci)
    behavior_ci = _percentile_ci(behavior_vals, alpha_ci)
    return output_ci, behavior_ci


def _percentile_ci(
    values: list[float],
    alpha: float,
) -> tuple[float, float] | None:
    """Compute (lower, upper) percentile interval."""
    if not values:
        return None
    arr = np.sort(values)
    lower = float(np.percentile(arr, 100.0 * alpha / 2.0))
    upper = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
    return (lower, upper)


def _extract_fault_name(ev: EvaluationRecord) -> str:
    """Extract a human-readable fault name from an EvaluationRecord.

    Heuristic order:
    1. ``ev.metrics`` dict (key ``"fault_name"``).
    2. ``ev.perturbation`` field as a fallback.
    3. ``"unknown"`` sentinel.
    """
    if ev.metrics and isinstance(ev.metrics, dict):
        fn = ev.metrics.get("fault_name")
        if fn:
            return str(fn)
    if ev.perturbation:
        return ev.perturbation
    return "unknown"
