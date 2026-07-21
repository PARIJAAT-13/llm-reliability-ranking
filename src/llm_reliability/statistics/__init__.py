"""
Statistical Analysis Engine Module.

Provides summary statistics, correlations, hypothesis testing, effect sizes,
confidence interval estimations, and ranking divergence analysis for comparative
analysis of LLM reliability.
"""

from llm_reliability.statistics.statistical_engine import StatisticalEngine
from llm_reliability.statistics.correlation import (
    compute_kendall_tau,
    compute_spearman,
)
from llm_reliability.statistics.hypothesis_tests import (
    run_paired_t_test,
    run_wilcoxon_test,
)
from llm_reliability.statistics.effect_sizes import (
    compute_cohens_d,
    compute_rank_biserial,
    compute_cliffs_delta,
)
from llm_reliability.statistics.confidence_intervals import compute_bootstrap_ci
from llm_reliability.statistics.assumptions import check_normality
from llm_reliability.statistics.result_models import (
    CorrelationResult,
    HypothesisTestResult,
    EffectSizeResult,
    ConfidenceIntervalResult,
    SummaryStatistics,
    StatisticalReport,
)
from llm_reliability.statistics.ranking_divergence import (
    RankingDivergenceResult,
    analyze_ranking_divergence,
    compute_ranking_overlap,
    compute_ranking_divergence,
    compute_rank_displacement,
)

__all__ = [
    "StatisticalEngine",
    "compute_kendall_tau",
    "compute_spearman",
    "run_paired_t_test",
    "run_wilcoxon_test",
    "compute_cohens_d",
    "compute_rank_biserial",
    "compute_cliffs_delta",
    "compute_bootstrap_ci",
    "check_normality",
    "CorrelationResult",
    "HypothesisTestResult",
    "EffectSizeResult",
    "ConfidenceIntervalResult",
    "SummaryStatistics",
    "StatisticalReport",
    "RankingDivergenceResult",
    "analyze_ranking_divergence",
    "compute_ranking_overlap",
    "compute_ranking_divergence",
    "compute_rank_displacement",
]
