"""
Distribution visualizations for reliability and success metrics.

Purpose
-------
Produce KDE plots, histograms, box plots, and violin plots that show the
spread and shape of score distributions across agents.

Responsibilities
----------------
- Histogram + KDE overlay for success and reliability score distributions
- Box plots comparing agents side-by-side across metric dimensions
- Violin plots for full distribution shapes
- Scatter plots pairing success vs. reliability scores

Usage example
-------------
>>> from llm_reliability.visualization.distributions import DistributionPlotter
>>> from llm_reliability.records.metric import MetricRecord
>>> plotter = DistributionPlotter()
>>> fig = plotter.plot_score_histogram(metrics, metric="success_rate")

How figures are produced
------------------------
Each method extracts the requested metric field from ``MetricRecord`` objects,
builds a Pandas Series, then uses seaborn (with matplotlib fallback) to render
the chart.  All axes share the publication style set in ``styles.py``.
"""

from __future__ import annotations

from typing import Any

from llm_reliability.visualization.plotter import BasePlotter
from llm_reliability.visualization.styles import (
    COLOR_RELIABILITY,
    COLOR_SUCCESS,
    FIG_HEIGHT_DEFAULT,
    FIG_WIDTH_DOUBLE,
    FONT_SIZE_LABEL,
    FONT_SIZE_TITLE,
    PALETTE,
)

_METRIC_DISPLAY: dict[str, str] = {
    "success_rate": "Success Rate",
    "composite_reliability": "Composite Reliability",
    "repeated_run_consistency": "Repeated-Run Consistency",
    "perturbation_robustness": "Perturbation Robustness",
    "fault_tolerance": "Fault Tolerance",
}


def _extract_scores(metrics: list[Any], metric: str) -> tuple[list[str], list[float]]:
    """Return (agent_names, scores) for the requested metric field."""
    agents, scores = [], []
    for m in metrics:
        val = getattr(m, metric, None)
        if val is not None:
            agents.append(m.agent)
            scores.append(float(val))
    return agents, scores


class DistributionPlotter(BasePlotter):
    """Generate distribution plots for reliability and success metrics.

    Parameters
    ----------
    figsize : tuple[float, float], optional
        Override default figure size.
    """

    def plot(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        """Dispatch to ``plot_score_histogram`` (default plot method)."""
        return self.plot_score_histogram(*args, **kwargs)

    # ------------------------------------------------------------------
    # Histogram / KDE
    # ------------------------------------------------------------------

    def plot_score_histogram(
        self,
        metrics: list[Any],
        metric: str = "success_rate",
        bins: int = 15,
        title: str | None = None,
        color: str | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Any:
        """Plot histogram with KDE overlay for one metric across agents.

        Parameters
        ----------
        metrics : list[MetricRecord]
            Metric records to visualise.
        metric : str
            Field name on ``MetricRecord`` to plot.
        bins : int
            Number of histogram bins.
        title : str, optional
            Figure title; auto-generated from *metric* if omitted.
        color : str, optional
            Bar colour.
        figsize : tuple[float, float], optional
            Override figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import numpy as np

        agents, scores = _extract_scores(metrics, metric)
        if not scores:
            fig, ax = self._new_figure(figsize=figsize)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
            return fig

        display_name = _METRIC_DISPLAY.get(metric, metric.replace("_", " ").title())
        title = title or f"Distribution of {display_name}"
        color = color or (COLOR_SUCCESS if "success" in metric else COLOR_RELIABILITY)

        fig, ax = self._new_figure(figsize=figsize)
        arr = np.array(scores)

        ax.hist(
            arr, bins=bins, color=color, alpha=0.7, edgecolor="white", linewidth=0.5, density=True
        )

        try:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(arr)
            x_range = np.linspace(arr.min() - 0.05, arr.max() + 0.05, 200)
            ax.plot(x_range, kde(x_range), color=color, linewidth=1.5, label="KDE")
            ax.legend()
        except ImportError:
            pass

        ax.set_xlabel(display_name, fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel("Density", fontsize=FONT_SIZE_LABEL)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE)
        ax.set_xlim([-0.05, 1.05])
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Box plots
    # ------------------------------------------------------------------

    def plot_box(
        self,
        metrics: list[Any],
        metric_fields: list[str] | None = None,
        title: str = "Metric Distributions",
        figsize: tuple[float, float] | None = None,
    ) -> Any:
        """Box plot comparing multiple metric dimensions across all agents.

        Parameters
        ----------
        metrics : list[MetricRecord]
            Metric records to plot.
        metric_fields : list[str], optional
            Which fields to include; defaults to all five reliability metrics.
        title : str
            Figure title.
        figsize : tuple[float, float], optional
            Override figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if metric_fields is None:
            metric_fields = [
                "success_rate",
                "composite_reliability",
                "repeated_run_consistency",
            ]

        data: dict[str, list[float]] = {}
        for field in metric_fields:
            _, scores = _extract_scores(metrics, field)
            if scores:
                data[_METRIC_DISPLAY.get(field, field)] = scores

        if not data:
            fig, ax = self._new_figure(figsize=figsize)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
            return fig

        labels = list(data.keys())
        values = [data[k] for k in labels]

        fig, ax = self._new_figure(figsize=figsize or (FIG_WIDTH_DOUBLE, FIG_HEIGHT_DEFAULT))
        bp = ax.boxplot(
            values,
            patch_artist=True,
            widths=0.5,
            medianprops={"color": "black", "linewidth": 1.5},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.6},
        )
        for patch, color in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(title, fontsize=FONT_SIZE_TITLE)
        ax.set_ylabel("Score", fontsize=FONT_SIZE_LABEL)
        ax.set_ylim([-0.05, 1.05])
        ax.set_xticklabels(labels, rotation=15, ha="right")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Violin plots
    # ------------------------------------------------------------------

    def plot_violin(
        self,
        metrics: list[Any],
        metric_fields: list[str] | None = None,
        title: str = "Score Distributions (Violin)",
        figsize: tuple[float, float] | None = None,
    ) -> Any:
        """Violin plot showing full distribution shape per metric.

        Parameters
        ----------
        metrics : list[MetricRecord]
            Metric records to visualise.
        metric_fields : list[str], optional
            Which fields to include.
        title : str
            Figure title.
        figsize : tuple[float, float], optional
            Override figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if metric_fields is None:
            metric_fields = ["success_rate", "composite_reliability", "repeated_run_consistency"]

        data: dict[str, list[float]] = {}
        for field in metric_fields:
            _, scores = _extract_scores(metrics, field)
            if scores:
                data[_METRIC_DISPLAY.get(field, field)] = scores

        if not data:
            fig, ax = self._new_figure(figsize=figsize)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
            return fig

        try:
            import pandas as pd
            import seaborn as sns

            rows = []
            for label, vals in data.items():
                for v in vals:
                    rows.append({"Metric": label, "Score": v})
            df = pd.DataFrame(rows)

            fig, ax = self._new_figure(figsize=figsize or (FIG_WIDTH_DOUBLE, FIG_HEIGHT_DEFAULT))
            sns.violinplot(
                data=df,
                x="Metric",
                y="Score",
                ax=ax,
                palette=PALETTE[: len(data)],
                inner="box",
                linewidth=0.8,
            )
            ax.set_title(title, fontsize=FONT_SIZE_TITLE)
            ax.set_xlabel("", fontsize=FONT_SIZE_LABEL)
            ax.set_ylabel("Score", fontsize=FONT_SIZE_LABEL)
            ax.set_ylim([-0.05, 1.05])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
            fig.tight_layout()
            return fig

        except ImportError:
            return self.plot_box(metrics, metric_fields, title, figsize)

    # ------------------------------------------------------------------
    # Scatter: success vs. reliability
    # ------------------------------------------------------------------

    def plot_scatter_success_vs_reliability(
        self,
        metrics: list[Any],
        title: str = "Success Rate vs. Composite Reliability",
        annotate: bool = True,
        figsize: tuple[float, float] | None = None,
    ) -> Any:
        """Scatter plot of success rate (x) vs. composite reliability (y).

        Agents that appear far from the diagonal have divergent rankings.

        Parameters
        ----------
        metrics : list[MetricRecord]
            Metric records.
        title : str
            Figure title.
        annotate : bool
            Whether to label each point with the agent name.
        figsize : tuple[float, float], optional
            Override figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import numpy as np

        if not metrics:
            fig, ax = self._new_figure(figsize=figsize)
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
            return fig

        fig, ax = self._new_figure(figsize=figsize)
        for i, m in enumerate(metrics):
            color = PALETTE[i % len(PALETTE)]
            ax.scatter(
                m.success_rate, m.composite_reliability, color=color, s=40, zorder=3, label=m.agent
            )
            if annotate:
                ax.annotate(
                    m.agent,
                    (m.success_rate, m.composite_reliability),
                    textcoords="offset points",
                    xytext=(4, 3),
                    fontsize=7,
                    color=color,
                )

        # Diagonal reference line y = x
        lim = np.linspace(0, 1, 100)
        ax.plot(lim, lim, "--", color="#aaaaaa", linewidth=0.8, label="y = x")

        ax.set_xlabel("Success Rate", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel("Composite Reliability", fontsize=FONT_SIZE_LABEL)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.legend(loc="lower right", fontsize=7)
        fig.tight_layout()
        return fig
