"""
Visualization package for the LLM Reliability Ranking framework.

Public API
----------
Styles
    ``apply_publication_style`` — apply IEEE-quality rcParams globally.
    ``PALETTE`` — colourblind-safe 8-colour palette.

Plotters
    ``BasePlotter`` — abstract base with save/export helpers.
    ``HeatmapPlotter`` — annotated correlation heatmaps.
    ``DistributionPlotter`` — histograms, box plots, violin plots, scatter.
    ``RankingPlotter`` — bar charts, rank divergence, bump charts.

Tables
    ``TableGenerator`` — DataFrames from MetricRecord / StatisticalReport.

Export
    ``FigureExporter`` — PNG, SVG, PDF.
    ``TableExporter`` — CSV, Excel, JSON, Markdown, LaTeX.
"""

from llm_reliability.visualization.styles import (
    apply_publication_style,
    PALETTE,
    COLOR_SUCCESS,
    COLOR_RELIABILITY,
    COLOR_WEIGHTED,
)
from llm_reliability.visualization.plotter import BasePlotter
from llm_reliability.visualization.heatmaps import HeatmapPlotter
from llm_reliability.visualization.distributions import DistributionPlotter
from llm_reliability.visualization.ranking_plots import RankingPlotter
from llm_reliability.visualization.tables import TableGenerator
from llm_reliability.visualization.export import FigureExporter, TableExporter

__all__ = [
    "apply_publication_style",
    "PALETTE",
    "COLOR_SUCCESS",
    "COLOR_RELIABILITY",
    "COLOR_WEIGHTED",
    "BasePlotter",
    "HeatmapPlotter",
    "DistributionPlotter",
    "RankingPlotter",
    "TableGenerator",
    "FigureExporter",
    "TableExporter",
]
