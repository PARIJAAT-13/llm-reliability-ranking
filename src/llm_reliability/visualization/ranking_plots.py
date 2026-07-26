"""
Ranking chart visualizations.

Purpose
-------
Produce horizontal bar charts, rank comparison scatter plots, and bump charts
that make ranking results immediately legible to a conference audience.

Responsibilities
----------------
- Horizontal bar chart for success rankings
- Horizontal bar chart for reliability rankings
- Side-by-side comparison bar chart
- Rank difference scatter plot (success rank vs. reliability rank)
- Bump chart showing rank trajectories across multiple ranking types

Usage example
-------------
>>> from llm_reliability.visualization.ranking_plots import RankingPlotter
>>> plotter = RankingPlotter()
>>> fig = plotter.plot_ranking_bar(ranking_record, title="Success Rankings")

How figures are produced
------------------------
Each method parses a ``RankingRecord`` (or list thereof) to extract ordered
``(agent, score)`` tuples, then builds the requested chart using matplotlib.
Colours are drawn from the publication palette in ``styles.py``.
"""

from __future__ import annotations

from typing import Any

from llm_reliability.visualization.plotter import BasePlotter
from llm_reliability.visualization.styles import (COLOR_RELIABILITY,
                                                  COLOR_SUCCESS,
                                                  COLOR_WEIGHTED,
                                                  FIG_HEIGHT_DEFAULT,
                                                  FIG_WIDTH_DOUBLE,
                                                  FONT_SIZE_ANNOTATION,
                                                  FONT_SIZE_LABEL,
                                                  FONT_SIZE_TICK,
                                                  FONT_SIZE_TITLE, PALETTE)


def _ranking_to_lists(ranking: Any) -> tuple[list[str], list[float], list[int]]:
    """Extract (agents, scores, ranks) from a RankingRecord."""
    agents = [r[0] for r in ranking.rankings]
    scores = [r[1] for r in ranking.rankings]
    ranks = [ranking.rank_map[a] for a in agents]
    return agents, scores, ranks


class RankingPlotter(BasePlotter):
    """Generate ranking charts from ``RankingRecord`` objects."""

    def plot(self, *args: Any, **kwargs: Any) -> Any:
        """Default: delegate to ``plot_ranking_bar``."""
        return self.plot_ranking_bar(*args, **kwargs)

    # ------------------------------------------------------------------
    # Horizontal bar chart
    # ------------------------------------------------------------------

    def plot_ranking_bar(
        self,
        ranking: Any,
        title: str | None = None,
        color: str | None = None,
        figsize: tuple[float, float] | None = None,
        show_values: bool = True,
    ) -> Any:
        """Horizontal bar chart: one bar per agent, sorted by rank.

        Parameters
        ----------
        ranking : RankingRecord
            The ranking to visualise.
        title : str, optional
            Figure title; auto-generated from ranking type if omitted.
        color : str, optional
            Bar colour; defaults to semantic colour by ranking type.
        figsize : tuple[float, float], optional
            Override figure size.
        show_values : bool
            Whether to annotate each bar with its score.

        Returns
        -------
        matplotlib.figure.Figure
        """
        agents, scores, _ = _ranking_to_lists(ranking)

        if not agents:
            fig, ax = self._new_figure(figsize=figsize)
            ax.text(
                0.5,
                0.5,
                "No rankings available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        rtype = getattr(ranking, "ranking_type", "success")
        if color is None:
            color = {
                "success": COLOR_SUCCESS,
                "reliability": COLOR_RELIABILITY,
                "weighted": COLOR_WEIGHTED,
            }.get(rtype, COLOR_SUCCESS)

        title = title or f"{rtype.title()} Rankings — {ranking.benchmark}"

        n = len(agents)
        fig_h = max(FIG_HEIGHT_DEFAULT, 0.4 * n)
        fig, ax = self._new_figure(figsize=figsize or (FIG_WIDTH_DOUBLE * 0.7, fig_h))

        y_pos = list(range(n - 1, -1, -1))  # highest rank at top
        bars = ax.barh(y_pos, scores, height=0.6, color=color, alpha=0.85, edgecolor="white")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(agents, fontsize=FONT_SIZE_TICK)
        ax.set_xlabel("Score", fontsize=FONT_SIZE_LABEL)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE)
        ax.set_xlim([0, 1.12])
        ax.axvline(x=0, color="#333333", linewidth=0.5)

        if show_values:
            for bar, score in zip(bars, scores):
                ax.text(
                    bar.get_width() + 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{score:.3f}",
                    va="center",
                    ha="left",
                    fontsize=FONT_SIZE_ANNOTATION,
                )

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Comparison bar chart (success vs. reliability side-by-side)
    # ------------------------------------------------------------------

    def plot_ranking_comparison(
        self,
        success_ranking: Any,
        reliability_ranking: Any,
        title: str = "Success vs. Reliability Rankings",
        figsize: tuple[float, float] | None = None,
    ) -> Any:
        """Side-by-side grouped bar chart comparing two ranking types.

        Parameters
        ----------
        success_ranking : RankingRecord
            Success-based ranking.
        reliability_ranking : RankingRecord
            Reliability-based ranking.
        title : str
            Figure title.
        figsize : tuple[float, float], optional
            Override figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import numpy as np

        s_agents, s_scores, _ = _ranking_to_lists(success_ranking)
        r_agents, r_scores, _ = _ranking_to_lists(reliability_ranking)

        # Use success ranking order as the canonical order
        all_agents = s_agents or r_agents
        s_map = {a: v for a, v in zip(s_agents, s_scores)}
        r_map = {a: v for a, v in zip(r_agents, r_scores)}

        s_vals = [s_map.get(a, 0.0) for a in all_agents]
        r_vals = [r_map.get(a, 0.0) for a in all_agents]

        n = len(all_agents)
        fig_h = max(FIG_HEIGHT_DEFAULT, 0.5 * n)
        fig, ax = self._new_figure(figsize=figsize or (FIG_WIDTH_DOUBLE, fig_h))

        y = np.arange(n)
        bar_h = 0.35
        ax.barh(
            y + bar_h / 2,
            s_vals,
            height=bar_h,
            color=COLOR_SUCCESS,
            alpha=0.85,
            label="Success",
            edgecolor="white",
        )
        ax.barh(
            y - bar_h / 2,
            r_vals,
            height=bar_h,
            color=COLOR_RELIABILITY,
            alpha=0.85,
            label="Reliability",
            edgecolor="white",
        )

        ax.set_yticks(y)
        ax.set_yticklabels(all_agents, fontsize=FONT_SIZE_TICK)
        ax.set_xlabel("Score", fontsize=FONT_SIZE_LABEL)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE)
        ax.set_xlim([0, 1.12])
        ax.legend(loc="lower right", fontsize=FONT_SIZE_ANNOTATION)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Rank divergence scatter
    # ------------------------------------------------------------------

    def plot_rank_divergence(
        self,
        success_ranking: Any,
        reliability_ranking: Any,
        title: str = "Rank Divergence: Success vs. Reliability",
        figsize: tuple[float, float] | None = None,
    ) -> Any:
        """Scatter of success rank (x) vs. reliability rank (y) per agent.

        Agents on the diagonal have consistent rankings.  Agents far off
        the diagonal indicate conditions where the two ranking types diverge.

        Parameters
        ----------
        success_ranking : RankingRecord
            Success-based ranking.
        reliability_ranking : RankingRecord
            Reliability-based ranking.
        title : str
            Figure title.
        figsize : tuple[float, float], optional
            Override figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import numpy as np

        s_map = dict(success_ranking.rank_map)
        r_map = dict(reliability_ranking.rank_map)
        common = [a for a in s_map if a in r_map]

        if not common:
            fig, ax = self._new_figure(figsize=figsize)
            ax.text(
                0.5,
                0.5,
                "No common agents",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        s_ranks = [s_map[a] for a in common]
        r_ranks = [r_map[a] for a in common]
        diffs = [abs(s - r) for s, r in zip(s_ranks, r_ranks)]
        max(diffs) if diffs else 1

        fig, ax = self._new_figure(figsize=figsize)

        for i, agent in enumerate(common):
            color = PALETTE[i % len(PALETTE)]
            ax.scatter(s_ranks[i], r_ranks[i], s=60, color=color, zorder=3)
            ax.annotate(
                agent,
                (s_ranks[i], r_ranks[i]),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=FONT_SIZE_ANNOTATION,
                color=color,
            )

        n_agents = len(common)
        diag = np.arange(1, n_agents + 2)
        ax.plot(diag, diag, "--", color="#aaaaaa", linewidth=0.8, label="No divergence")

        ax.set_xlabel("Success Rank", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel("Reliability Rank", fontsize=FONT_SIZE_LABEL)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.legend(fontsize=FONT_SIZE_ANNOTATION)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Bump chart
    # ------------------------------------------------------------------

    def plot_bump_chart(
        self,
        rankings: list[Any],
        ranking_labels: list[str] | None = None,
        title: str = "Rank Trajectories Across Ranking Types",
        figsize: tuple[float, float] | None = None,
    ) -> Any:
        """Bump chart showing how agents' ranks change across ranking types.

        Parameters
        ----------
        rankings : list[RankingRecord]
            Ordered list of rankings (e.g. [success_ranking, reliability_ranking]).
        ranking_labels : list[str], optional
            x-axis labels for each ranking; defaults to ranking_type field.
        title : str
            Figure title.
        figsize : tuple[float, float], optional
            Override figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not rankings:
            fig, ax = self._new_figure(figsize=figsize)
            ax.text(
                0.5,
                0.5,
                "No rankings provided",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        labels = ranking_labels or [
            getattr(r, "ranking_type", str(i)) for i, r in enumerate(rankings)
        ]
        all_agents = sorted({a for r in rankings for a in r.rank_map})

        fig, ax = self._new_figure(
            figsize=figsize
            or (FIG_WIDTH_DOUBLE * 0.8, max(FIG_HEIGHT_DEFAULT, 0.45 * len(all_agents)))
        )

        for idx, agent in enumerate(all_agents):
            color = PALETTE[idx % len(PALETTE)]
            x_vals = list(range(len(rankings)))
            y_vals = [r.rank_map.get(agent, float("nan")) for r in rankings]

            ax.plot(
                x_vals,
                y_vals,
                "-o",
                color=color,
                linewidth=1.2,
                markersize=5,
                label=agent,
                zorder=3,
            )

            # Annotate left endpoint
            if y_vals[0] == y_vals[0]:  # not nan
                ax.annotate(
                    agent,
                    (0, y_vals[0]),
                    textcoords="offset points",
                    xytext=(-4, 0),
                    ha="right",
                    fontsize=FONT_SIZE_ANNOTATION,
                    color=color,
                )

        ax.set_xticks(range(len(rankings)))
        ax.set_xticklabels(labels, fontsize=FONT_SIZE_TICK)
        ax.set_ylabel("Rank (1 = best)", fontsize=FONT_SIZE_LABEL)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE)
        ax.invert_yaxis()
        ax.set_yticks(range(1, len(all_agents) + 1))
        ax.legend(loc="upper right", fontsize=FONT_SIZE_ANNOTATION, ncol=2)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Experiment timeline
    # ------------------------------------------------------------------

    def plot_experiment_timeline(
        self,
        execution_records: list[Any],
        title: str = "Experiment Timeline",
        figsize: tuple[float, float] | None = None,
    ) -> Any:
        """Gantt-style timeline of execution records.

        Parameters
        ----------
        execution_records : list[ExecutionRecord]
            Records with ``started_at`` and ``completed_at`` attributes.
        title : str
            Figure title.
        figsize : tuple[float, float], optional
            Override figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from datetime import datetime

        sorted_records = sorted(
            execution_records,
            key=lambda r: r.timestamp,
        )

        if not sorted_records:
            fig, ax = self._new_figure(figsize=figsize)
            ax.text(
                0.5,
                0.5,
                "No timeline data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        fig_h = max(FIG_HEIGHT_DEFAULT, 0.35 * len(sorted_records))
        fig, ax = self._new_figure(figsize=figsize or (FIG_WIDTH_DOUBLE, fig_h))

        base_ts = datetime.fromisoformat(sorted_records[0].timestamp)
        for i, rec in enumerate(sorted_records):
            try:
                t_start = datetime.fromisoformat(rec.timestamp)
                duration = rec.runtime_seconds
                start_offset = (t_start - base_ts).total_seconds()
                label = f"{rec.agent}@{rec.benchmark}"
                color = PALETTE[i % len(PALETTE)]
                ax.barh(
                    i,
                    duration,
                    left=start_offset,
                    height=0.6,
                    color=color,
                    alpha=0.8,
                    edgecolor="white",
                )
                ax.text(
                    start_offset + 0.5,
                    i,
                    label,
                    va="center",
                    fontsize=FONT_SIZE_ANNOTATION,
                )
            except Exception:
                continue

        ax.set_xlabel("Elapsed Time (s)", fontsize=FONT_SIZE_LABEL)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE)
        ax.set_yticks([])
        fig.tight_layout()
        return fig
