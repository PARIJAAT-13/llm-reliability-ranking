"""
Statistical Analysis Engine.

Orchestrates all statistical computations on RankingRecords.
"""

from typing import Any

from llm_reliability.records.ranking import RankingRecord
from llm_reliability.statistics.correlation import (
    compute_kendall_tau,
    compute_spearman,
    _align_ranking_scores,
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
from llm_reliability.statistics.utils import (
    validate_rankings,
    calculate_summary_statistics,
)
from llm_reliability.statistics.result_models import (
    CorrelationResult,
    HypothesisTestResult,
    EffectSizeResult,
    ConfidenceIntervalResult,
    SummaryStatistics,
    StatisticalReport,
)


class StatisticalEngine:
    """Orchestrates statistical analyses comparing agent performance rankings."""

    @staticmethod
    def compute_correlations(
        ranking1: RankingRecord,
        ranking2: RankingRecord,
    ) -> dict[str, CorrelationResult]:
        """Compute Spearman and Kendall Tau correlations between two rankings.

        Parameters
        ----------
        ranking1 : RankingRecord
            The first ranking.
        ranking2 : RankingRecord
            The second ranking.

        Returns
        -------
        dict[str, CorrelationResult]
            Dictionary of computed correlations.
        """
        validate_rankings(ranking1, ranking2)
        return {
            "spearman": compute_spearman(ranking1, ranking2),
            "kendall_tau": compute_kendall_tau(ranking1, ranking2),
        }

    @staticmethod
    def compute_significance(
        ranking1: RankingRecord,
        ranking2: RankingRecord,
    ) -> list[HypothesisTestResult]:
        """Run significance tests (Paired t-test and Wilcoxon) between rankings.

        Parameters
        ----------
        ranking1 : RankingRecord
            The first ranking.
        ranking2 : RankingRecord
            The second ranking.

        Returns
        -------
        list[HypothesisTestResult]
            List of hypothesis test results.
        """
        validate_rankings(ranking1, ranking2)
        return [
            run_paired_t_test(ranking1, ranking2),
            run_wilcoxon_test(ranking1, ranking2),
        ]

    @staticmethod
    def compute_effect_sizes(
        ranking1: RankingRecord,
        ranking2: RankingRecord,
    ) -> list[EffectSizeResult]:
        """Compute Cohen's d, Rank-biserial correlation, and Cliff's Delta.

        Parameters
        ----------
        ranking1 : RankingRecord
            The first ranking.
        ranking2 : RankingRecord
            The second ranking.

        Returns
        -------
        list[EffectSizeResult]
            List of effect size results.
        """
        validate_rankings(ranking1, ranking2)
        return [
            compute_cohens_d(ranking1, ranking2),
            compute_rank_biserial(ranking1, ranking2),
            compute_cliffs_delta(ranking1, ranking2),
        ]

    @staticmethod
    def compute_confidence_intervals(
        ranking1: RankingRecord,
        ranking2: RankingRecord,
        confidence_level: float = 0.95,
    ) -> dict[str, ConfidenceIntervalResult]:
        """Compute bootstrap confidence intervals for scores and differences.

        Parameters
        ----------
        ranking1 : RankingRecord
            The first ranking.
        ranking2 : RankingRecord
            The second ranking.
        confidence_level : float, default 0.95
            The confidence interval level.

        Returns
        -------
        dict[str, ConfidenceIntervalResult]
            Confidence interval results.
        """
        validate_rankings(ranking1, ranking2)
        x, y = _align_ranking_scores(ranking1, ranking2)
        diffs = [a - b for a, b in zip(x, y)]

        return {
            "ranking1": compute_bootstrap_ci(x, confidence_level=confidence_level),
            "ranking2": compute_bootstrap_ci(y, confidence_level=confidence_level),
            "differences": compute_bootstrap_ci(diffs, confidence_level=confidence_level),
        }

    @staticmethod
    def compute_summary_statistics(
        ranking1: RankingRecord,
        ranking2: RankingRecord,
    ) -> dict[str, SummaryStatistics]:
        """Generate summary statistics for rankings and differences.

        Parameters
        ----------
        ranking1 : RankingRecord
            The first ranking.
        ranking2 : RankingRecord
            The second ranking.

        Returns
        -------
        dict[str, SummaryStatistics]
            Summary statistics mapped by group name.
        """
        validate_rankings(ranking1, ranking2)
        x, y = _align_ranking_scores(ranking1, ranking2)
        diffs = [a - b for a, b in zip(x, y)]

        return {
            "ranking1": calculate_summary_statistics(x),
            "ranking2": calculate_summary_statistics(y),
            "differences": calculate_summary_statistics(diffs),
        }

    @classmethod
    def analyze(
        cls,
        ranking1: RankingRecord,
        ranking2: RankingRecord,
        confidence_level: float = 0.95,
    ) -> StatisticalReport:
        """Perform a complete statistical comparative analysis between two rankings.

        Parameters
        ----------
        ranking1 : RankingRecord
            The first ranking.
        ranking2 : RankingRecord
            The second ranking.
        confidence_level : float, default 0.95
            The bootstrap confidence level.

        Returns
        -------
        StatisticalReport
            Pydantic model containing all analysis results.
        """
        validate_rankings(ranking1, ranking2)
        
        sum_stats = cls.compute_summary_statistics(ranking1, ranking2)
        corrs = cls.compute_correlations(ranking1, ranking2)
        tests = cls.compute_significance(ranking1, ranking2)
        effects = cls.compute_effect_sizes(ranking1, ranking2)
        cis = cls.compute_confidence_intervals(ranking1, ranking2, confidence_level)

        metadata = {
            "sample_size": len(ranking1.rankings),
            "ranking1_type": ranking1.ranking_type,
            "ranking2_type": ranking2.ranking_type,
            "benchmark": ranking1.benchmark,
        }

        return StatisticalReport(
            summary_statistics=sum_stats,
            correlations=corrs,
            hypothesis_tests=tests,
            effect_sizes=effects,
            confidence_intervals=cis,
            metadata=metadata,
        )
