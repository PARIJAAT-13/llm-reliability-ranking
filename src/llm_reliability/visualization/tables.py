"""
Table generation for publication-quality result summaries.

Purpose
-------
Convert experiment records and statistical outputs into Pandas DataFrames
and formatted table strings (Markdown, LaTeX, CSV, JSON).

Responsibilities
----------------
- Summary statistics table from MetricRecords
- Reliability metrics table (all five dimensions per agent)
- Ranking table comparing success vs. reliability ranks
- Hypothesis test results table
- Effect size table
- Confidence interval table
- Benchmark summary table
- Agent summary table

Usage example
-------------
>>> from llm_reliability.visualization.tables import TableGenerator
>>> gen = TableGenerator()
>>> df = gen.reliability_metrics_table(metrics)
>>> print(df.to_markdown(index=False))

How tables are produced
-----------------------
Each method builds a list of row dictionaries from the input records,
constructs a Pandas DataFrame, and returns it.  Callers may then use
``TableExporter`` (in ``export.py``) to persist the DataFrame to disk.
"""

from __future__ import annotations

from typing import Any


class TableGenerator:
    """Produces Pandas DataFrames from experiment records.

    All methods return a ``pandas.DataFrame`` so callers can further filter,
    sort, or style the data before export.
    """

    # ------------------------------------------------------------------
    # Metric / reliability tables
    # ------------------------------------------------------------------

    def summary_statistics_table(
        self,
        metrics: list[Any],
        benchmark: str | None = None,
    ) -> Any:
        """Summary statistics table (mean, std, min, max per metric dimension).

        Parameters
        ----------
        metrics : list[MetricRecord]
            Records to summarise.
        benchmark : str, optional
            Filter to a specific benchmark; includes all if ``None``.

        Returns
        -------
        pandas.DataFrame
            Columns: Benchmark, Agent, Evaluations, Success Rate, Consistency,
            Perturbation Robustness, Fault Tolerance, Composite Reliability.
        """
        import pandas as pd

        if benchmark:
            metrics = [m for m in metrics if m.benchmark == benchmark]

        rows = []
        for m in metrics:
            rows.append(
                {
                    "Benchmark": m.benchmark,
                    "Agent": m.agent,
                    "Evaluations": m.evaluation_count,
                    "Success Rate": round(m.success_rate, 4),
                    "Consistency": round(m.repeated_run_consistency, 4),
                    "Pert. Robustness": (
                        round(m.perturbation_robustness, 4)
                        if m.perturbation_robustness is not None
                        else None
                    ),
                    "Fault Tolerance": (
                        round(m.fault_tolerance, 4) if m.fault_tolerance is not None else None
                    ),
                    "Composite Reliability": round(m.composite_reliability, 4),
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Sort by benchmark then composite reliability descending
        df = df.sort_values(
            ["Benchmark", "Composite Reliability"],
            ascending=[True, False],
        ).reset_index(drop=True)
        return df

    def reliability_metrics_table(self, metrics: list[Any]) -> Any:
        """Detailed reliability metrics per agent.

        Parameters
        ----------
        metrics : list[MetricRecord]

        Returns
        -------
        pandas.DataFrame
        """
        return self.summary_statistics_table(metrics)

    # ------------------------------------------------------------------
    # Ranking tables
    # ------------------------------------------------------------------

    def ranking_table(
        self,
        success_ranking: Any,
        reliability_ranking: Any,
    ) -> Any:
        """Combined ranking table: success rank, reliability rank, rank difference.

        Parameters
        ----------
        success_ranking : RankingRecord
        reliability_ranking : RankingRecord

        Returns
        -------
        pandas.DataFrame
            Columns: Agent, Success Rank, Success Score, Reliability Rank,
            Reliability Score, Rank Δ.
        """
        import pandas as pd

        s_map = dict(success_ranking.rank_map)
        r_map = dict(reliability_ranking.rank_map)
        s_scores = {a: sc for a, sc in success_ranking.rankings}
        r_scores = {a: sc for a, sc in reliability_ranking.rankings}

        all_agents = sorted(set(s_map) | set(r_map))
        rows = []
        for agent in all_agents:
            s_rank = s_map.get(agent)
            r_rank = r_map.get(agent)
            delta = (s_rank - r_rank) if (s_rank is not None and r_rank is not None) else None
            rows.append(
                {
                    "Agent": agent,
                    "Success Rank": s_rank,
                    "Success Score": round(s_scores.get(agent, float("nan")), 4),
                    "Reliability Rank": r_rank,
                    "Reliability Score": round(r_scores.get(agent, float("nan")), 4),
                    "Rank Δ": delta,
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("Success Rank").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Statistical result tables
    # ------------------------------------------------------------------

    def hypothesis_test_table(self, statistical_report: Any) -> Any:
        """Hypothesis test results table from a ``StatisticalReport``.

        Parameters
        ----------
        statistical_report : StatisticalReport

        Returns
        -------
        pandas.DataFrame
            Columns: Test, Statistic, p-value, Significant (α=0.05),
            Assumptions Met, Warnings.
        """
        import pandas as pd

        rows = []
        for result in statistical_report.hypothesis_tests:
            rows.append(
                {
                    "Test": result.method,
                    "Statistic": round(result.statistic, 6),
                    "p-value": round(result.p_value, 6),
                    "Significant (α=0.05)": result.p_value < 0.05,
                    "Assumptions Met": result.assumptions_met,
                    "Warnings": "; ".join(result.warnings) if result.warnings else "",
                }
            )
        return pd.DataFrame(rows)

    def effect_size_table(self, statistical_report: Any) -> Any:
        """Effect size table from a ``StatisticalReport``.

        Parameters
        ----------
        statistical_report : StatisticalReport

        Returns
        -------
        pandas.DataFrame
            Columns: Method, Value, Interpretation.
        """
        import pandas as pd

        rows = []
        for result in statistical_report.effect_sizes:
            rows.append(
                {
                    "Method": result.method,
                    "Value": round(result.value, 6),
                    "Interpretation": result.interpretation,
                }
            )
        return pd.DataFrame(rows)

    def confidence_interval_table(self, statistical_report: Any) -> Any:
        """Confidence interval table from a ``StatisticalReport``.

        Parameters
        ----------
        statistical_report : StatisticalReport

        Returns
        -------
        pandas.DataFrame
            Columns: Variable, Lower, Upper, Level.
        """
        import pandas as pd

        rows = []
        for name, ci in statistical_report.confidence_intervals.items():
            rows.append(
                {
                    "Variable": name,
                    "Lower": round(ci.lower, 6),
                    "Upper": round(ci.upper, 6),
                    "Confidence Level": ci.confidence_level,
                }
            )
        return pd.DataFrame(rows)

    def correlation_table(self, statistical_report: Any) -> Any:
        """Correlation coefficient table from a ``StatisticalReport``.

        Parameters
        ----------
        statistical_report : StatisticalReport

        Returns
        -------
        pandas.DataFrame
            Columns: Method, Coefficient, p-value.
        """
        import pandas as pd

        rows = []
        for name, result in statistical_report.correlations.items():
            rows.append(
                {
                    "Method": result.method,
                    "Coefficient": round(result.coefficient, 6),
                    "p-value": round(result.p_value, 6),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Benchmark / agent summary tables
    # ------------------------------------------------------------------

    def benchmark_summary_table(self, metrics: list[Any]) -> Any:
        """Aggregate statistics grouped by benchmark.

        Parameters
        ----------
        metrics : list[MetricRecord]

        Returns
        -------
        pandas.DataFrame
            Columns: Benchmark, # Agents, Mean Success, Mean Reliability, …
        """
        from collections import defaultdict

        import numpy as np
        import pandas as pd

        groups: dict[str, list[Any]] = defaultdict(list)
        for m in metrics:
            groups[m.benchmark].append(m)

        rows = []
        for bench, mlist in groups.items():
            rows.append(
                {
                    "Benchmark": bench,
                    "# Agents": len(mlist),
                    "Mean Success Rate": round(np.mean([m.success_rate for m in mlist]), 4),
                    "Std Success Rate": round(np.std([m.success_rate for m in mlist]), 4),
                    "Mean Reliability": round(np.mean([m.composite_reliability for m in mlist]), 4),
                    "Std Reliability": round(np.std([m.composite_reliability for m in mlist]), 4),
                    "Total Evaluations": sum(m.evaluation_count for m in mlist),
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("Benchmark").reset_index(drop=True)
        return df

    def agent_summary_table(self, metrics: list[Any]) -> Any:
        """Aggregate statistics grouped by agent.

        Parameters
        ----------
        metrics : list[MetricRecord]

        Returns
        -------
        pandas.DataFrame
            Columns: Agent, # Benchmarks, Mean Success, Mean Reliability, …
        """
        from collections import defaultdict

        import numpy as np
        import pandas as pd

        groups: dict[str, list[Any]] = defaultdict(list)
        for m in metrics:
            groups[m.agent].append(m)

        rows = []
        for agent, mlist in groups.items():
            rows.append(
                {
                    "Agent": agent,
                    "# Benchmarks": len(mlist),
                    "Mean Success Rate": round(np.mean([m.success_rate for m in mlist]), 4),
                    "Mean Consistency": round(
                        np.mean([m.repeated_run_consistency for m in mlist]), 4
                    ),
                    "Mean Reliability": round(np.mean([m.composite_reliability for m in mlist]), 4),
                    "Total Evaluations": sum(m.evaluation_count for m in mlist),
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("Mean Reliability", ascending=False).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def to_latex(
        self,
        df: Any,
        caption: str = "",
        label: str = "tab:results",
        index: bool = False,
        float_format: str = "%.4f",
    ) -> str:
        """Render a DataFrame as a LaTeX ``tabular`` environment.

        Parameters
        ----------
        df : pd.DataFrame
            Input table.
        caption : str
            Table caption.
        label : str
            LaTeX label for ``\\ref{}``.
        index : bool
            Whether to include the index.
        float_format : str
            Format string for float columns.

        Returns
        -------
        str
            LaTeX table string.
        """
        try:
            latex = df.to_latex(
                index=index,
                caption=caption,
                label=label,
                float_format=float_format,
                escape=True,
            )
            return latex
        except Exception:
            return df.to_string(index=index)

    def to_markdown(self, df: Any, index: bool = False) -> str:
        """Render a DataFrame as a Markdown table.

        Parameters
        ----------
        df : pd.DataFrame
        index : bool

        Returns
        -------
        str
        """
        try:
            return df.to_markdown(index=index)
        except Exception:
            return df.to_string(index=index)
