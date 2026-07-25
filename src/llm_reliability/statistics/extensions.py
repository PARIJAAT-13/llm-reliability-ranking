"""
Extended statistical analysis suite for LLM reliability research.

Adds to the existing :mod:`llm_reliability.statistics` package:

1. **More hypothesis tests**: Mann-Whitney U, Kruskal-Wallis H, Friedman test,
   one-way ANOVA, post-hoc Nemenyi test.
2. **More effect sizes**: Hedges' g (bias-corrected), Glass's Delta, Eta-squared,
   Omega-squared.
3. **Multiple comparison correction**: Bonferroni, Holm-Bonferroni,
   Benjamini-Hochberg (FDR).
4. **Power analysis**: Post-hoc and a-priori power calculations.
5. **Bayesian analysis**: Bayes factor for the paired t-test (optional,
   using an approximation when ``scipy`` is available).

All functions accept raw sequences (``list[float]``) or numpy arrays, making
them usable both standalone and within the existing
:class:`~llm_reliability.statistics.statistical_engine.StatisticalEngine`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

from llm_reliability.statistics.result_models import (
    EffectSizeResult,
    HypothesisTestResult,
)

# ======================================================================
# 1. Multiple Comparison Correction
# ======================================================================


def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Apply Bonferroni correction for multiple comparisons.

    Parameters
    ----------
    p_values:
        Raw p-values from multiple hypothesis tests.
    alpha:
        Family-wise error rate (default 0.05).

    Returns
    -------
    list[bool]
        Boolean array where ``True`` indicates significance after correction.
    """
    n = len(p_values)
    if n == 0:
        return []
    corrected_alpha = alpha / n
    return [p <= corrected_alpha for p in p_values]


def holm_bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Apply Holm-Bonferroni sequential correction (more powerful than Bonferroni).

    Parameters
    ----------
    p_values:
        Raw p-values from multiple hypothesis tests.
    alpha:
        Family-wise error rate (default 0.05).

    Returns
    -------
    list[bool]
        Boolean array indicating significance after correction, maintaining
        original order of *p_values*.
    """
    n = len(p_values)
    if n == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    rejected = [False] * n
    for k, (orig_idx, p) in enumerate(indexed):
        threshold = alpha / (n - k)
        if p <= threshold:
            rejected[orig_idx] = True
        else:
            break
    return rejected


def benjamini_hochberg_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Apply Benjamini-Hochberg procedure to control FDR.

    Parameters
    ----------
    p_values:
        Raw p-values from multiple hypothesis tests.
    alpha:
        Desired false discovery rate (default 0.05).

    Returns
    -------
    list[bool]
        Boolean array indicating significance after correction.
    """
    n = len(p_values)
    if n == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    rejected = [False] * n
    max_k = -1
    for k, (orig_idx, p) in enumerate(indexed):
        threshold = (k + 1) / n * alpha
        if p <= threshold:
            max_k = k
    for k in range(max_k + 1):
        orig_idx = indexed[k][0]
        rejected[orig_idx] = True
    return rejected


# ======================================================================
# 2. Additional Hypothesis Tests
# ======================================================================


def run_mannwhitney_u(
    sample1: list[float],
    sample2: list[float],
    alternative: str = "two-sided",
) -> HypothesisTestResult:
    """Mann-Whitney U test for independent samples.

    Non-parametric alternative to the independent t-test.
    """
    n1, n2 = len(sample1), len(sample2)
    warnings: list[str] = []
    if n1 < 3 or n2 < 3:
        warnings.append("Very small samples (n<3) — results may be unreliable.")

    if n1 == 0 or n2 == 0:
        return HypothesisTestResult(
            statistic=0.0,
            p_value=1.0,
            method="Mann-Whitney U",
            alternative=alternative,
            assumptions_met=True,
            warnings=warnings,
        )

    stat, p = scipy_stats.mannwhitneyu(sample1, sample2, alternative=alternative)
    return HypothesisTestResult(
        statistic=float(stat),
        p_value=float(p),
        method="Mann-Whitney U",
        alternative=alternative,
        assumptions_met=True,
        warnings=warnings,
    )


def run_kruskal_wallis(
    *samples: list[float],
) -> HypothesisTestResult:
    """Kruskal-Wallis H-test for independent samples (3+ groups).

    Non-parametric alternative to one-way ANOVA.
    """
    valid = [s for s in samples if len(s) > 0]
    if len(valid) < 2:
        return HypothesisTestResult(
            statistic=0.0,
            p_value=1.0,
            method="Kruskal-Wallis H",
            alternative="two-sided",
            assumptions_met=True,
            warnings=["Fewer than 2 non-empty groups."],
        )

    warnings: list[str] = []
    for i, s in enumerate(valid):
        if len(s) < 3:
            warnings.append(f"Group {i} has fewer than 3 observations.")

    stat, p = scipy_stats.kruskal(*valid)
    return HypothesisTestResult(
        statistic=float(stat),
        p_value=float(p),
        method="Kruskal-Wallis H",
        alternative="two-sided",
        assumptions_met=True,
        warnings=warnings,
    )


def run_friedman_test(
    *samples: list[float],
) -> HypothesisTestResult:
    """Friedman test for repeated-measures (3+ conditions).

    Non-parametric alternative to repeated-measures ANOVA.
    """
    valid = [np.array(s, dtype=float) for s in samples if len(s) > 0]
    if len(valid) < 3:
        return HypothesisTestResult(
            statistic=0.0,
            p_value=1.0,
            method="Friedman Test",
            alternative="two-sided",
            assumptions_met=True,
            warnings=["Fewer than 3 conditions."],
        )

    n_obs = len(valid[0])
    for i, s in enumerate(valid):
        if len(s) != n_obs:
            valid[i] = np.pad(s, (0, n_obs - len(s)), constant_values=np.nan)

    with np.errstate(invalid="ignore"):
        valid_arr = np.column_stack(valid)
        mask = ~np.any(np.isnan(valid_arr), axis=1)
        valid_arr = valid_arr[mask]

    if valid_arr.shape[0] < 2:
        return HypothesisTestResult(
            statistic=0.0,
            p_value=1.0,
            method="Friedman Test",
            alternative="two-sided",
            assumptions_met=True,
            warnings=["Not enough complete observations."],
        )

    stat, p = scipy_stats.friedmanchisquare(*[valid_arr[:, i] for i in range(valid_arr.shape[1])])
    p_val = float(p) if not math.isnan(p) else 1.0
    return HypothesisTestResult(
        statistic=float(stat) if not math.isnan(stat) else 0.0,
        p_value=p_val,
        method="Friedman Test",
        alternative="two-sided",
        assumptions_met=True,
    )


def run_oneway_anova(
    *samples: list[float],
) -> HypothesisTestResult:
    """One-way ANOVA for independent samples (parametric).

    Assumes normality and homoscedasticity.
    """
    valid = [np.array(s, dtype=float) for s in samples if len(s) > 0]
    if len(valid) < 2:
        return HypothesisTestResult(
            statistic=0.0,
            p_value=1.0,
            method="One-way ANOVA",
            alternative="two-sided",
            assumptions_met=False,
            warnings=["Fewer than 2 groups."],
        )

    warnings: list[str] = []
    for i, s in enumerate(valid):
        if len(s) < 3:
            warnings.append(
                f"Group {i} has fewer than 3 observations — normality check unreliable."
            )

    # Shapiro-Wilk normality check for each group
    assumptions_met = True
    for i, s in enumerate(valid):
        if len(s) >= 3:
            _, p_norm = scipy_stats.shapiro(s)
            if p_norm < 0.05:
                assumptions_met = False
                warnings.append(f"Group {i} violates normality (Shapiro-Wilk p={p_norm:.4f}).")

    if not assumptions_met:
        warnings.append("Normality violated — consider Kruskal-Wallis instead.")

    stat, p = scipy_stats.f_oneway(*valid)
    return HypothesisTestResult(
        statistic=float(stat),
        p_value=float(p),
        method="One-way ANOVA",
        alternative="two-sided",
        assumptions_met=assumptions_met,
        warnings=warnings,
    )


def run_nemenyi_posthoc(*samples: list[float]) -> list[HypothesisTestResult]:
    """Post-hoc Nemenyi test for pairwise comparisons after Kruskal-Wallis.

    Parameters
    ----------
    samples:
        Two or more sample groups.

    Returns
    -------
    list[HypothesisTestResult]
        Pairwise comparison results with corrected p-values.
    """
    valid = [np.array(s, dtype=float) for s in samples if len(s) > 0]
    k = len(valid)
    if k < 2:
        return []

    # Compute overall ranks
    all_data = np.concatenate(valid)
    ranked = scipy_stats.rankdata(all_data)

    # Split ranks back by group
    split_ranks: list[np.ndarray] = []
    idx = 0
    for arr in valid:
        split_ranks.append(ranked[idx : idx + len(arr)])
        idx += len(arr)

    group_means = np.array([np.mean(r) for r in split_ranks])
    n_total = len(all_data)

    # Critical difference based on Nemenyi: CD = q_alpha * sqrt(k(k+1)/(6n))
    # Studentized range statistic q for k groups, infinite df
    from scipy.stats import studentized_range

    results: list[HypothesisTestResult] = []
    for i in range(k):
        for j in range(i + 1, k):
            # Compute test statistic (Nemenyi uses studentized range)
            se = math.sqrt(k * (k + 1) / (6 * n_total))
            stat_val = abs(group_means[i] - group_means[j]) / se

            # Approximate p-value using studentized range distribution
            try:
                p_val = 1.0 - studentized_range.cdf(stat_val * math.sqrt(2), k, float("inf"))
            except Exception:
                p_val = 1.0

            results.append(
                HypothesisTestResult(
                    statistic=float(stat_val),
                    p_value=float(p_val),
                    method="Nemenyi Post-hoc",
                    alternative="two-sided",
                    assumptions_met=True,
                    warnings=[],
                )
            )

    return results


# ======================================================================
# 3. Additional Effect Sizes
# ======================================================================


_EFFECT_SIZE_THRESHOLDS: dict[str, list[tuple[float, str]]] = {
    "cohens_d": [(0.2, "negligible"), (0.5, "small"), (0.8, "medium"), (float("inf"), "large")],
    "hedges_g": [(0.2, "negligible"), (0.5, "small"), (0.8, "medium"), (float("inf"), "large")],
    "glasss_delta": [(0.2, "negligible"), (0.5, "small"), (0.8, "medium"), (float("inf"), "large")],
    "eta_squared": [
        (0.01, "negligible"),
        (0.06, "small"),
        (0.14, "medium"),
        (float("inf"), "large"),
    ],
    "omega_squared": [
        (0.01, "negligible"),
        (0.06, "small"),
        (0.14, "medium"),
        (float("inf"), "large"),
    ],
}


def _interpret(value: float, metric: str) -> str:
    """Return qualitative interpretation for an effect size."""
    thresholds = _EFFECT_SIZE_THRESHOLDS.get(metric, [])
    for threshold, label in thresholds:
        if abs(value) < threshold:
            return label
    return "very large"


def compute_hedges_g(
    sample1: list[float],
    sample2: list[float],
) -> EffectSizeResult:
    """Hedges' g — bias-corrected Cohen's d for unequal sample sizes.

    Corrects for small-sample bias using the c(n) correction factor.
    """
    n1, n2 = len(sample1), len(sample2)
    if n1 < 2 or n2 < 2:
        return EffectSizeResult(
            value=0.0,
            method="Hedges' g",
            interpretation="insufficient data",
        )

    mean1, mean2 = float(np.mean(sample1)), float(np.mean(sample2))
    var1, var2 = float(np.var(sample1, ddof=1)), float(np.var(sample2, ddof=1))

    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1.0

    d = (mean1 - mean2) / pooled_std

    # Hedges' correction factor: c(n) = 1 - 3/(4*(n1+n2) - 9)
    correction = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    g = d * correction

    interpretation = _interpret(g, "hedges_g")
    return EffectSizeResult(value=float(g), method="Hedges' g", interpretation=interpretation)


def compute_glasss_delta(
    sample1: list[float],
    sample2: list[float],
    control_group: str = "sample1",
) -> EffectSizeResult:
    """Glass's Delta — effect size using the control group's standard deviation.

    Useful when the treatment may affect variance.
    """
    if control_group == "sample1":
        control_std = float(np.std(sample1, ddof=1)) if len(sample1) > 1 else 1.0
    else:
        control_std = float(np.std(sample2, ddof=1)) if len(sample2) > 1 else 1.0

    if len(sample1) < 1 or len(sample2) < 1:
        return EffectSizeResult(
            value=0.0, method="Glass's Delta", interpretation="insufficient data"
        )

    mean1, mean2 = float(np.mean(sample1)), float(np.mean(sample2))
    delta = (mean1 - mean2) / control_std if control_std > 0 else 0.0

    interpretation = _interpret(delta, "glasss_delta")
    return EffectSizeResult(
        value=float(delta),
        method="Glass's Delta",
        interpretation=interpretation,
    )


def compute_eta_squared(
    *samples: list[float],
) -> EffectSizeResult:
    """Eta-squared (η²) for ANOVA models — proportion of variance explained.

    Accepts 2+ groups; computes eta-squared = SS_between / SS_total.
    """
    valid = [np.array(s, dtype=float) for s in samples if len(s) > 0]
    if len(valid) < 2:
        return EffectSizeResult(value=0.0, method="Eta-squared", interpretation="insufficient data")

    all_data = np.concatenate(valid)
    grand_mean = float(np.mean(all_data))

    ss_between = sum(len(g) * (float(np.mean(g)) - grand_mean) ** 2 for g in valid)
    ss_within = sum(float(np.sum((g - float(np.mean(g))) ** 2)) for g in valid)
    ss_total = ss_between + ss_within

    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
    interpretation = _interpret(eta_sq, "eta_squared")
    return EffectSizeResult(
        value=float(eta_sq),
        method="Eta-squared",
        interpretation=interpretation,
    )


def compute_omega_squared(
    *samples: list[float],
) -> EffectSizeResult:
    """Omega-squared (ω²) — unbiased proportion of variance explained.

    Less biased than eta-squared for small samples.
    ω² = (SS_between - (k-1) * MS_within) / (SS_total + MS_within)
    """
    valid = [np.array(s, dtype=float) for s in samples if len(s) > 0]
    k = len(valid)
    if k < 2:
        return EffectSizeResult(
            value=0.0, method="Omega-squared", interpretation="insufficient data"
        )

    all_data = np.concatenate(valid)
    n_total = len(all_data)
    grand_mean = float(np.mean(all_data))

    ss_between = sum(len(g) * (float(np.mean(g)) - grand_mean) ** 2 for g in valid)
    ss_within = sum(float(np.sum((g - float(np.mean(g))) ** 2)) for g in valid)
    ss_total = ss_between + ss_within

    df_between = k - 1
    df_within = n_total - k
    ms_within = ss_within / df_within if df_within > 0 else 0.0

    omega_sq = (
        (ss_between - df_between * ms_within) / (ss_total + ms_within)
        if (ss_total + ms_within) > 0
        else 0.0
    )
    omega_sq = max(0.0, omega_sq)

    interpretation = _interpret(omega_sq, "omega_squared")
    return EffectSizeResult(
        value=float(omega_sq),
        method="Omega-squared",
        interpretation=interpretation,
    )


# ======================================================================
# 4. Power Analysis
# ======================================================================


def compute_posthoc_power(
    d: float,
    n: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """Estimate statistical power for a one-sample / paired t-test.

    Uses the non-central t-distribution approximation.

    Parameters
    ----------
    d:
        Cohen's d effect size.
    n:
        Sample size.
    alpha:
        Significance level (default 0.05).
    alternative:
        ``"two-sided"`` or ``"one-sided"``.

    Returns
    -------
    float
        Estimated power in [0, 1].
    """
    if n < 2 or d == 0.0:
        return 0.0

    df = n - 1
    ncp = d * math.sqrt(n)

    if alternative == "two-sided":
        t_crit = scipy_stats.t.ppf(1.0 - alpha / 2, df)
        power = 1.0 - scipy_stats.nct.cdf(t_crit, df, ncp) + scipy_stats.nct.cdf(-t_crit, df, ncp)
    else:
        t_crit = scipy_stats.t.ppf(1.0 - alpha, df)
        power = 1.0 - scipy_stats.nct.cdf(t_crit, df, ncp)

    return float(np.clip(power, 0.0, 1.0))


def compute_required_sample_size(
    d: float,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> int:
    """Estimate minimum sample size required to achieve a given power.

    Uses iterative search over the non-central t-distribution.

    Parameters
    ----------
    d:
        Expected Cohen's d effect size.
    power:
        Desired statistical power (default 0.8).
    alpha:
        Significance level (default 0.05).
    alternative:
        ``"two-sided"`` or ``"one-sided"``.

    Returns
    -------
    int
        Minimum sample size needed.
    """
    if d <= 0:
        return 10000  # Effect size too small to detect

    for n in range(3, 10001):
        p = compute_posthoc_power(d, n, alpha, alternative)
        if p >= power:
            return n
    return 10000


# ======================================================================
# 5. Bayesian Analysis (Approximation)
# ======================================================================


def compute_bayes_factor_ttest(
    sample1: list[float],
    sample2: list[float],
) -> dict[str, Any]:
    """Approximate Bayes factor BF10 for a paired t-test.

    Uses the BIC approximation:
        BF10 ≈ exp((BIC_H0 - BIC_H1) / 2)

    where BIC_H1 assumes a unit-information prior on the effect size.

    Parameters
    ----------
    sample1:
        First paired sample.
    sample2:
        Second paired sample.

    Returns
    -------
    dict with keys:
        - ``bf10`` — Bayes factor for H1 over H0.
        - ``log_bf`` — natural log of BF10.
        - ``interpretation`` — qualitative evidence category.
        - ``n`` — sample size.
        - ``d`` — Cohen's d.
    """
    if len(sample1) != len(sample2) or len(sample1) < 3:
        return {
            "bf10": 1.0,
            "log_bf": 0.0,
            "interpretation": "insufficient data",
            "n": min(len(sample1), len(sample2)),
            "d": 0.0,
        }

    differences = np.array(sample1, dtype=float) - np.array(sample2, dtype=float)
    n = len(differences)
    d = float(np.mean(differences)) / (float(np.std(differences, ddof=1)) + 1e-10)

    # BIC approximation for Bayesian t-test (Rouder et al., 2009)
    # BIC_H0 = n * ln(SS_res / n) + k0 * ln(n)  where k0=1
    # BIC_H1 = n * ln(SS_res' / n) + k1 * ln(n)  where k1=2
    # For a paired t-test: t^2 = n * d^2
    # BF10 ≈ exp((BIC_H0 - BIC_H1) / 2)
    t_stat = d * math.sqrt(n)
    bic_h0 = n * math.log(1.0 / (1.0 + t_stat**2 / n) + 1e-10) + 1 * math.log(n)
    bic_h1 = n * math.log(1.0 + 1e-10)
    log_bf = (bic_h0 - bic_h1) / 2.0
    bf10 = math.exp(log_bf)

    # Interpretation (Kass & Raftery, 1995)
    if bf10 > 100:
        interpretation = "decisive evidence for H1"
    elif bf10 > 30:
        interpretation = "strong evidence for H1"
    elif bf10 > 10:
        interpretation = "substantial evidence for H1"
    elif bf10 > 3:
        interpretation = "moderate evidence for H1"
    elif bf10 > 1:
        interpretation = "anecdotal evidence for H1"
    elif bf10 > 0.33:
        interpretation = "moderate evidence for H0"
    elif bf10 > 0.1:
        interpretation = "substantial evidence for H0"
    else:
        interpretation = "strong evidence for H0"

    return {
        "bf10": float(bf10),
        "log_bf": float(log_bf),
        "interpretation": interpretation,
        "n": n,
        "d": float(d),
    }
