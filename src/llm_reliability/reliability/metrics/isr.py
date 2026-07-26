from __future__ import annotations

from collections import defaultdict

import numpy as np

from llm_reliability.records.evaluation import EvaluationRecord

DEFAULT_N_BINS: int = 10
DEFAULT_ISR_ALPHA: float = 0.6
DEFAULT_N_RESAMPLES: int = 1000
DEFAULT_CI_ALPHA: float = 0.05
DEFAULT_N_WINDOWS: int = 5


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
    if not evaluations:
        raise ValueError("Cannot compute ISR from empty evaluations.")

    baseline = [ev for ev in evaluations if not ev.fault_injected]
    faulted = [ev for ev in evaluations if ev.fault_injected]

    if not faulted:
        raise ValueError("No fault-injected evaluations found. Cannot compute ISR.")

    sr_base = float(np.mean([ev.score for ev in baseline])) if baseline else 0.0
    sr_fault = float(np.mean([ev.score for ev in faulted]))
    isr_behavior = 1.0 - abs(sr_base - sr_fault)

    isr_output = _histogram_intersection(
        [ev.score for ev in baseline] if baseline else [],
        [ev.score for ev in faulted],
        n_bins=n_bins,
    )

    isr_composite = alpha * isr_output + (1.0 - alpha) * isr_behavior

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

    overall = compute_isr(evaluations, n_bins=n_bins, alpha=alpha)

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
        combined = baseline + window_evs
        win_result = compute_isr(combined, n_bins=n_bins, alpha=alpha)
        window_isr.append(win_result["isr_composite"])

    trend_slope = 0.0
    if len(window_isr) >= 2:
        x: np.ndarray = np.arange(len(window_isr), dtype=float)
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


def _histogram_intersection(
    baseline_scores: list[float],
    fault_scores: list[float],
    n_bins: int,
) -> float:
    if not fault_scores:
        return 1.0
    if not baseline_scores:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    base_hist, _ = np.histogram(baseline_scores, bins=bins, density=True)
    fault_hist, _ = np.histogram(fault_scores, bins=bins, density=True)

    intersection = float(np.sum(np.minimum(base_hist, fault_hist)))
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

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        b_hist, _ = np.histogram(b_sample, bins=bins, density=True)
        f_hist, _ = np.histogram(f_sample, bins=bins, density=True)
        intersection = float(np.sum(np.minimum(b_hist, f_hist)))
        bin_width = 1.0 / n_bins
        output_vals.append(intersection * bin_width)

        sr_b = float(np.mean(b_sample))
        sr_f = float(np.mean(f_sample))
        behavior_vals.append(1.0 - abs(sr_b - sr_f))

    output_ci = _percentile_ci(output_vals, alpha_ci)
    behavior_ci = _percentile_ci(behavior_vals, alpha_ci)
    return output_ci, behavior_ci


def _percentile_ci(
    values: list[float],
    alpha: float,
) -> tuple[float, float] | None:
    if not values:
        return None
    arr = np.sort(values)
    lower = float(np.percentile(arr, 100.0 * alpha / 2.0))
    upper = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
    return (lower, upper)


def _extract_fault_name(ev: EvaluationRecord) -> str:
    if ev.metrics and isinstance(ev.metrics, dict):
        fn = ev.metrics.get("fault_name")
        if fn:
            return str(fn)
    if ev.perturbation:
        return ev.perturbation
    return "unknown"
