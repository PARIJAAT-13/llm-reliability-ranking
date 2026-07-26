"""
Statistical Analysis Engine Module.

Provides summary statistics, correlations, hypothesis testing, effect sizes,
confidence interval estimations, ranking divergence analysis, multiple comparison
correction, power analysis, and Bayesian analysis for comparative analysis of
LLM reliability.
"""

from __future__ import annotations

from llm_reliability.statistics.assumptions import check_normality
from llm_reliability.statistics.confidence_intervals import compute_bootstrap_ci
from llm_reliability.statistics.correlation import compute_kendall_tau, compute_spearman
from llm_reliability.statistics.effect_sizes import (
    compute_cliffs_delta,
    compute_cohens_d,
    compute_rank_biserial,
)
from llm_reliability.statistics.extensions import (
    benjamini_hochberg_correction,
    bonferroni_correction,
    compute_bayes_factor_ttest,
    compute_eta_squared,
    compute_glasss_delta,
    compute_hedges_g,
    compute_omega_squared,
    compute_posthoc_power,
    compute_required_sample_size,
    holm_bonferroni_correction,
    run_friedman_test,
    run_kruskal_wallis,
    run_mannwhitney_u,
    run_nemenyi_posthoc,
    run_oneway_anova,
)
from llm_reliability.statistics.hypothesis_tests import (
    run_paired_t_test,
    run_wilcoxon_test,
)
from llm_reliability.statistics.ranking_divergence import (
    RankingDivergenceResult,
    analyze_ranking_divergence,
    compute_rank_displacement,
    compute_ranking_divergence,
    compute_ranking_overlap,
)
from llm_reliability.statistics.report import generate_reliability_statistical_report
from llm_reliability.statistics.result_models import (
    ConfidenceIntervalResult,
    CorrelationResult,
    EffectSizeResult,
    HypothesisTestResult,
    StatisticalReport,
    SummaryStatistics,
)
from llm_reliability.statistics.statistical_engine import StatisticalEngine

__all__ = [
    "StatisticalEngine",
    "compute_kendall_tau",
    "compute_spearman",
    "run_paired_t_test",
    "run_wilcoxon_test",
    "run_mannwhitney_u",
    "run_kruskal_wallis",
    "run_friedman_test",
    "run_oneway_anova",
    "run_nemenyi_posthoc",
    "compute_cohens_d",
    "compute_rank_biserial",
    "compute_cliffs_delta",
    "compute_hedges_g",
    "compute_glasss_delta",
    "compute_eta_squared",
    "compute_omega_squared",
    "compute_bootstrap_ci",
    "check_normality",
    "bonferroni_correction",
    "holm_bonferroni_correction",
    "benjamini_hochberg_correction",
    "compute_posthoc_power",
    "compute_required_sample_size",
    "compute_bayes_factor_ttest",
    "CorrelationResult",
    "HypothesisTestResult",
    "EffectSizeResult",
    "ConfidenceIntervalResult",
    "SummaryStatistics",
    "StatisticalReport",
    "generate_reliability_statistical_report",
    "RankingDivergenceResult",
    "analyze_ranking_divergence",
    "compute_ranking_overlap",
    "compute_ranking_divergence",
    "compute_rank_displacement",
]
