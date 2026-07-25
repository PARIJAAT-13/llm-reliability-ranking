"""
Automated statistical procedure selection for LLM reliability research.

Given data characteristics (normality, variance homogeneity, sample size,
number of groups), this module automatically recommends the appropriate:

- Hypothesis test
- Effect size measure
- Multiple comparison correction method

This addresses Gap G5 from the literature review: "Statistical rigor is
lacking — only 16 % of benchmark studies use statistical tests."

Decision tree
-------------
For **two groups**:
  - Normal + homogeneous variances → independent t-test (Welch if unequal)
  - Non-normal → Mann-Whitney U test
  - Paired design → paired t-test (normal) or Wilcoxon (non-normal)

For **three+ groups**:
  - Normal + homogeneous → one-way ANOVA
  - Normal + unequal variances → Welch ANOVA
  - Non-normal → Kruskal-Wallis H test
  - Repeated measures → repeated measures ANOVA (normal) / Friedman (non-normal)

Effect size selection
---------------------
  - t-test → Cohen's d (or Hedges' g for small n)
  - Mann-Whitney → rank-biserial correlation (r)
  - ANOVA → eta-squared (η²) or omega-squared (ω²)
  - Kruskal-Wallis → epsilon-squared (ε²)
  - Friedman → Kendall's W

Correction selection
--------------------
  - < 5 comparisons → Bonferroni (conservative)
  - 5–20 comparisons → Holm-Bonferroni (balanced)
  - 20+ comparisons → Benjamini-Hochberg (FDR, less conservative)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as scipy_stats

from llm_reliability.statistics.extensions import (
    compute_hedges_g,
    run_friedman_test,
    run_kruskal_wallis,
    run_mannwhitney_u,
    run_oneway_anova,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

TestSuggestion = dict[str, str | list[str]]
EffectSizeSuggestion = dict[str, str | float]
CorrectionSuggestion = dict[str, str | list[bool] | float]

# ---------------------------------------------------------------------------
# Assumption checks
# ---------------------------------------------------------------------------


def check_normality(
    samples: list[list[float]],
    alpha: float = 0.05,
) -> list[bool]:
    """Check normality for each sample using Shapiro-Wilk.

    Parameters
    ----------
    samples:
        List of groups, where each group is a list of scores.
    alpha:
        Significance threshold (default 0.05).

    Returns
    -------
    list[bool]
        ``True`` if the sample appears normal (p >= alpha).
    """
    results: list[bool] = []
    for group in samples:
        arr = np.asarray(group, dtype=float)
        if len(arr) < 3:
            results.append(True)
            continue
        if np.var(arr) == 0:
            results.append(True)
            continue
        try:
            _, p = scipy_stats.shapiro(arr)
            results.append(bool(p >= alpha))
        except Exception:
            results.append(True)
    return results


def check_variance_homogeneity(
    samples: list[list[float]],
    alpha: float = 0.05,
) -> bool:
    """Check homogeneity of variances using Levene's test.

    Parameters
    ----------
    samples:
        List of groups.
    alpha:
        Significance threshold (default 0.05).

    Returns
    -------
    bool
        ``True`` if variances appear homogeneous (p >= alpha).
    """
    if len(samples) < 2:
        return True
    filtered = [np.asarray(g, dtype=float) for g in samples if len(g) >= 2]
    if len(filtered) < 2:
        return True
    try:
        _, p = scipy_stats.levene(*filtered)
        return bool(p >= alpha)
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Test selection
# ---------------------------------------------------------------------------


def suggest_test(
    n_groups: int,
    is_paired: bool = False,
    is_normal: bool = True,
    is_homogeneous: bool = True,
) -> str:
    """Suggest the most appropriate hypothesis test.

    Parameters
    ----------
    n_groups:
        Number of groups / conditions (2 for two-sample, 3+ for multi-sample).
    is_paired:
        Whether the design is paired / repeated-measures.
    is_normal:
        Whether data appear normally distributed.
    is_homogeneous:
        Whether variances appear homogeneous (ignored for non-parametric).

    Returns
    -------
    str
        Recommended test name.
    """
    if n_groups == 2:
        if is_paired:
            return "paired_t_test" if is_normal else "wilcoxon_signed_rank"
        else:
            if is_normal:
                return "welch_t_test" if not is_homogeneous else "independent_t_test"
            return "mannwhitney_u"
    elif n_groups >= 3:
        if is_paired:
            return "repeated_measures_anova" if is_normal else "friedman_test"
        else:
            if is_normal and is_homogeneous:
                return "anova_oneway"
            elif is_normal and not is_homogeneous:
                return "welch_anova"
            return "kruskal_wallis"
    return "no_test"


def suggest_posthoc(
    n_groups: int,
    is_normal: bool,
    is_paired: bool = False,
) -> str | None:
    """Suggest an appropriate post-hoc test.

    Parameters
    ----------
    n_groups:
        Number of groups.
    is_normal:
        Whether data appear normal.
    is_paired:
        Whether design is paired.

    Returns
    -------
    str or None
        Recommended post-hoc test, or None if n_groups == 2.
    """
    if n_groups <= 2:
        return None
    if is_paired:
        return "bonferroni_corrected_paired_t" if is_normal else "bonferroni_corrected_wilcoxon"
    return "tukey_hsd" if is_normal else "nemenyi_test"


# ---------------------------------------------------------------------------
# Effect size selection
# ---------------------------------------------------------------------------


def suggest_effect_size(test_name: str, n_total: int) -> str:
    """Suggest the most appropriate effect size for a given test.

    Parameters
    ----------
    test_name:
        Name of the hypothesis test.
    n_total:
        Total sample size (used to choose between Cohen's d and Hedges' g).

    Returns
    -------
    str
        Recommended effect size name.
    """
    mapping: dict[str, str] = {
        "independent_t_test": "hedges_g" if n_total < 30 else "cohens_d",
        "welch_t_test": "hedges_g",
        "paired_t_test": "hedges_g",
        "mannwhitney_u": "rank_biserial_r",
        "wilcoxon_signed_rank": "rank_biserial_r",
        "anova_oneway": "omega_squared",
        "welch_anova": "omega_squared",
        "repeated_measures_anova": "partial_eta_squared",
        "kruskal_wallis": "epsilon_squared",
        "friedman_test": "kendalls_w",
        "nemenyi_test": "effect_measure_not_applicable",
    }
    return mapping.get(test_name, "cohens_d")


# ---------------------------------------------------------------------------
# Correction selection
# ---------------------------------------------------------------------------


def suggest_correction(
    n_comparisons: int,
    alpha: float = 0.05,
) -> str:
    """Suggest the most appropriate multiple comparison correction.

    Parameters
    ----------
    n_comparisons:
        Number of pairwise comparisons.
    alpha:
        Family-wise error rate or FDR threshold.

    Returns
    -------
    str
        ``"bonferroni"``, ``"holm_bonferroni"``, or ``"benjamini_hochberg"``.
    """
    if n_comparisons <= 5:
        return "bonferroni"
    elif n_comparisons <= 20:
        return "holm_bonferroni"
    return "benjamini_hochberg"


# ---------------------------------------------------------------------------
# Full recommendation
# ---------------------------------------------------------------------------


def auto_select(
    samples: list[list[float]],
    *,
    is_paired: bool = False,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Automatically select test, effect size, and correction.

    Parameters
    ----------
    samples:
        List of groups, each a list of float scores.
    is_paired:
        Whether the design is paired/repeated-measures.
    alpha:
        Significance threshold for assumption checks and correction.

    Returns
    -------
    dict with keys:
        - ``n_groups``          — number of groups.
        - ``n_comparisons``     — number of pairwise comparisons.
        - ``is_normal``         — whether all groups appear normal.
        - ``is_homogeneous``    — whether variances appear homogeneous.
        - ``recommended_test``  — suggested test name.
        - ``recommended_posthoc`` — suggested post-hoc test (or None).
        - ``recommended_effect_size`` — suggested effect size name.
        - ``recommended_correction``  — suggested correction method.
        - ``normality_results`` — per-group normality boolean list.
    """
    n_groups = len(samples)
    n_total = sum(len(g) for g in samples)

    normality = check_normality(samples, alpha=alpha)
    is_normal = all(normality)
    is_homogeneous = check_variance_homogeneity(samples, alpha=alpha)

    recommended_test = suggest_test(
        n_groups=n_groups,
        is_paired=is_paired,
        is_normal=is_normal,
        is_homogeneous=is_homogeneous,
    )

    recommended_posthoc = suggest_posthoc(
        n_groups=n_groups,
        is_normal=is_normal,
        is_paired=is_paired,
    )

    recommended_effect_size = suggest_effect_size(recommended_test, n_total)

    n_pairs = n_groups * (n_groups - 1) // 2 if n_groups >= 2 else 0
    recommended_correction = suggest_correction(n_pairs, alpha=alpha)

    return {
        "n_groups": n_groups,
        "n_comparisons": n_pairs,
        "is_normal": is_normal,
        "is_homogeneous": is_homogeneous,
        "recommended_test": recommended_test,
        "recommended_posthoc": recommended_posthoc,
        "recommended_effect_size": recommended_effect_size,
        "recommended_correction": recommended_correction,
        "normality_results": normality,
    }


# ======================================================================
# Convenience: run recommended test
# ======================================================================


def run_recommended_test(
    samples: list[list[float]],
    *,
    is_paired: bool = False,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Run the auto-selected hypothesis test and return results.

    Parameters
    ----------
    samples:
        List of groups.
    is_paired:
        Whether design is paired.
    alpha:
        Significance level.

    Returns
    -------
    dict with keys:
        - ``recommendation``  — output of ``auto_select()``.
        - ``test_name``       — name of test that was run.
        - ``statistic``       — test statistic value.
        - ``p_value``         — p-value.
        - ``is_significant``  — whether p < alpha.
        - ``effect_sizes``    — computed effect size(s) where applicable.
    """
    recommendation = auto_select(
        samples,
        is_paired=is_paired,
        alpha=alpha,
    )
    test_name = recommendation["recommended_test"]

    statistic: float = 0.0
    p_value: float = 1.0
    effect_sizes: dict[str, float] = {}

    if len(samples) < 2:
        return {
            "recommendation": recommendation,
            "test_name": "no_test",
            "statistic": 0.0,
            "p_value": 1.0,
            "is_significant": False,
            "error": "Need at least 2 groups.",
        }

    try:
        if test_name == "independent_t_test":
            from scipy.stats import ttest_ind

            stat, p = ttest_ind(samples[0], samples[1])
            statistic, p_value = float(stat), float(p)
            g = compute_hedges_g(samples[0], samples[1])
            effect_sizes = {"hedges_g": g.value}

        elif test_name == "welch_t_test":
            from scipy.stats import ttest_ind

            stat, p = ttest_ind(samples[0], samples[1], equal_var=False)
            statistic, p_value = float(stat), float(p)
            g = compute_hedges_g(samples[0], samples[1])
            effect_sizes = {"hedges_g": g.value}

        elif test_name == "mannwhitney_u":
            mw = run_mannwhitney_u(samples[0], samples[1])
            statistic, p_value = mw.statistic, mw.p_value
            n1, n2 = len(samples[0]), len(samples[1])
            r = 1.0 - 2.0 * statistic / (n1 * n2) if n1 * n2 > 0 else 0.0
            effect_sizes = {"rank_biserial_r": r}

        elif test_name == "paired_t_test":
            from scipy.stats import ttest_rel

            stat, p = ttest_rel(samples[0], samples[1])
            statistic, p_value = float(stat), float(p)
            g = compute_hedges_g(samples[0], samples[1])
            effect_sizes = {"hedges_g": g.value}

        elif test_name == "wilcoxon_signed_rank":
            from scipy.stats import wilcoxon

            stat, p = wilcoxon(samples[0], samples[1])
            statistic, p_value = float(stat), float(p)
            n = len(samples[0])
            r = float(stat / (n * (n + 1) / 2)) if n > 0 else 0.0
            effect_sizes = {"rank_biserial_r": r}

        elif test_name == "anova_oneway":
            aov = run_oneway_anova(*samples)
            statistic, p_value = aov.statistic, aov.p_value
            effect_sizes = {
                "eta_squared": (
                    statistic / (statistic + sum(len(g) for g in samples) - len(samples))
                    if (statistic + sum(len(g) for g in samples) - len(samples)) > 0
                    else 0.0
                )
            }

        elif test_name == "kruskal_wallis":
            kw = run_kruskal_wallis(*samples)
            statistic, p_value = kw.statistic, kw.p_value
            n_total = sum(len(g) for g in samples)
            effect_sizes = {
                "eta_squared": (
                    (statistic - len(samples) + 1) / (n_total - len(samples))
                    if n_total > len(samples)
                    else 0.0
                )
            }

        elif test_name == "friedman_test":
            fr = run_friedman_test(*samples)
            statistic, p_value = fr.statistic, fr.p_value

        else:
            return {
                "recommendation": recommendation,
                "test_name": test_name,
                "statistic": 0.0,
                "p_value": 1.0,
                "is_significant": False,
                "error": f"Automatic execution not implemented for '{test_name}'. Run manually.",
            }

    except Exception as e:
        return {
            "recommendation": recommendation,
            "test_name": test_name,
            "statistic": 0.0,
            "p_value": 1.0,
            "is_significant": False,
            "error": str(e),
        }

    is_significant = p_value < alpha

    return {
        "recommendation": recommendation,
        "test_name": test_name,
        "statistic": statistic,
        "p_value": p_value,
        "is_significant": is_significant,
        "effect_sizes": effect_sizes,
    }
