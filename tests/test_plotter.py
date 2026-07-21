"""Tests for figure generation and plotters."""

import os
import tempfile
import pathlib
import pytest
import matplotlib
matplotlib.use("Agg")

from llm_reliability.visualization.styles import apply_publication_style
from llm_reliability.visualization.plotter import BasePlotter
from llm_reliability.visualization.heatmaps import HeatmapPlotter
from llm_reliability.visualization.distributions import DistributionPlotter
from llm_reliability.visualization.ranking_plots import RankingPlotter
from llm_reliability.visualization.export import FigureExporter

from tests.ranking_test_helpers import create_mock_metric
from tests.statistics_test_helpers import create_mock_ranking
from llm_reliability.statistics.result_models import CorrelationResult, StatisticalReport, SummaryStatistics


class SimplePlotter(BasePlotter):
    """Simple concrete implementation of BasePlotter for testing."""
    def plot(self, data):
        fig, ax = self._new_figure()
        ax.plot(data)
        return fig


def test_base_plotter_save():
    plotter = SimplePlotter()
    fig = plotter.plot([1, 2, 3])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "test_plot"
        # Test saving in specific formats
        for fmt in ["png", "svg", "pdf"]:
            saved_path = plotter.save(fig, path, fmt=fmt, close=False)
            assert saved_path.exists()
            assert saved_path.suffix == f".{fmt}"
        
        # Test save_all
        paths = plotter.save_all(fig, path)
        assert len(paths) == 3
        for fmt in ["png", "svg", "pdf"]:
            assert paths[fmt].exists()


def test_heatmap_plotter():
    plotter = HeatmapPlotter()
    
    # Test plot matrix
    matrix = [[1.0, 0.5], [0.5, 1.0]]
    labels = ["Agent A", "Agent B"]
    fig = plotter.plot(matrix, labels=labels, title="Test Heatmap")
    assert fig is not None
    
    # Test plot from dict
    correlations = {
        "spearman": CorrelationResult(coefficient=0.8, p_value=0.01, method="Spearman"),
        "kendall_tau": CorrelationResult(coefficient=0.7, p_value=0.02, method="Kendall"),
    }
    fig_dict = plotter.plot_from_dict(correlations)
    assert fig_dict is not None
    
    # Test pairwise matrix
    fig_pw = plotter.plot_pairwise_matrix(labels, [[1.0, 0.8], [0.8, 1.0]])
    assert fig_pw is not None
    
    plotter.close_all()


def test_distribution_plotter():
    plotter = DistributionPlotter()
    
    metrics = [
        create_mock_metric("Agent A", success_rate=0.8, consistency=0.9),
        create_mock_metric("Agent B", success_rate=0.6, consistency=0.7),
    ]
    
    # Test plot score histogram
    fig_hist = plotter.plot_score_histogram(metrics, metric="success_rate")
    assert fig_hist is not None
    
    # Test plot box
    fig_box = plotter.plot_box(metrics)
    assert fig_box is not None
    
    # Test plot violin
    fig_violin = plotter.plot_violin(metrics)
    assert fig_violin is not None
    
    # Test scatter
    fig_scatter = plotter.plot_scatter_success_vs_reliability(metrics)
    assert fig_scatter is not None
    
    plotter.close_all()


def test_ranking_plotter():
    plotter = RankingPlotter()
    
    ranking_s = create_mock_ranking({"Agent A": 0.8, "Agent B": 0.6}, ranking_type="success")
    ranking_r = create_mock_ranking({"Agent A": 0.9, "Agent B": 0.7}, ranking_type="reliability")
    
    # Test ranking bar
    fig_bar = plotter.plot_ranking_bar(ranking_s)
    assert fig_bar is not None
    
    # Test ranking comparison
    fig_comp = plotter.plot_ranking_comparison(ranking_s, ranking_r)
    assert fig_comp is not None
    
    # Test rank divergence
    fig_div = plotter.plot_rank_divergence(ranking_s, ranking_r)
    assert fig_div is not None
    
    # Test bump chart
    fig_bump = plotter.plot_bump_chart([ranking_s, ranking_r])
    assert fig_bump is not None


def test_figure_exporter():
    fig = SimplePlotter().plot([1, 2, 3])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "exported_plot"
        # Test save_all on exporter
        paths = FigureExporter.save_all(fig, path)
        assert len(paths) == 3
        for fmt in ["png", "svg", "pdf"]:
            assert paths[fmt].exists()
