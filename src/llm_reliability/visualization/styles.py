"""
Publication-quality style configuration for all visualizations.

Purpose
-------
Centralises every aesthetic constant used by the visualization package so that
all figures share a coherent visual identity suitable for submission to an
international AI conference (NeurIPS / ICLR / ICML style).

Responsibilities
----------------
- Define colour palette (colourblind-safe, no rainbow)
- Set matplotlib rcParams for publication resolution and typography
- Expose helper ``apply_publication_style()`` that callers invoke once
- Provide named colours referenced throughout the package

Usage example
-------------
>>> from llm_reliability.visualization.styles import apply_publication_style, PALETTE
>>> apply_publication_style()
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

#: Colourblind-safe palette inspired by Wong (2011) / Paul Tol qualitative set.
PALETTE: Final[list[str]] = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#D55E00",  # vermilion
    "#F0E442",  # yellow
    "#000000",  # black
]

#: Semantic colour assignments.
COLOR_SUCCESS: Final[str] = PALETTE[0]  # blue  — success-based ranking
COLOR_RELIABILITY: Final[str] = PALETTE[1]  # orange — reliability-based ranking
COLOR_WEIGHTED: Final[str] = PALETTE[2]  # green  — weighted ranking
COLOR_DIVERGE_LOW: Final[str] = PALETTE[5]  # vermilion — negative divergence
COLOR_DIVERGE_HIGH: Final[str] = PALETTE[0]  # blue      — positive divergence
COLOR_NEUTRAL: Final[str] = "#888888"  # grey      — neutral / reference

#: Heatmap diverging colormap (does not rely on rainbow).
CMAP_DIVERGING: Final[str] = "RdBu_r"
CMAP_SEQUENTIAL: Final[str] = "Blues"

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------

FONT_FAMILY: Final[str] = "serif"
FONT_SIZE_BASE: Final[int] = 10
FONT_SIZE_TITLE: Final[int] = 11
FONT_SIZE_LABEL: Final[int] = 9
FONT_SIZE_TICK: Final[int] = 8
FONT_SIZE_LEGEND: Final[int] = 8
FONT_SIZE_ANNOTATION: Final[int] = 7

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

#: IEEE double-column paper: single-column width ≈ 3.5 in, double ≈ 7.16 in.
FIG_WIDTH_SINGLE: Final[float] = 3.5
FIG_WIDTH_DOUBLE: Final[float] = 7.16
FIG_HEIGHT_DEFAULT: Final[float] = 2.8

DPI_SCREEN: Final[int] = 100
DPI_PRINT: Final[int] = 300

LINE_WIDTH: Final[float] = 1.0
MARKER_SIZE: Final[float] = 4.0
SPINE_WIDTH: Final[float] = 0.5

# ---------------------------------------------------------------------------
# Grid and borders
# ---------------------------------------------------------------------------

GRID_ALPHA: Final[float] = 0.3
GRID_LINESTYLE: Final[str] = "--"
GRID_LINEWIDTH: Final[float] = 0.5

# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def apply_publication_style() -> None:
    """Apply publication-quality rcParams globally.

    Call this once at the top of any script or notebook that generates
    figures.  The style is designed to produce figures that meet the
    aesthetic standards of NeurIPS, ICLR, and ICML.

    How it works
    ------------
    The function updates ``matplotlib.rcParams`` in-place, so every
    subsequent figure inherits the settings.  It also calls
    ``seaborn.set_theme`` when seaborn is available.
    """
    try:
        import matplotlib as mpl

        mpl.rcParams.update(
            {
                # Figure
                "figure.dpi": DPI_SCREEN,
                "figure.figsize": [FIG_WIDTH_SINGLE, FIG_HEIGHT_DEFAULT],
                "figure.facecolor": "white",
                "figure.edgecolor": "white",
                # Axes
                "axes.facecolor": "white",
                "axes.edgecolor": "#333333",
                "axes.linewidth": SPINE_WIDTH,
                "axes.grid": True,
                "axes.grid.axis": "y",
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.titlesize": FONT_SIZE_TITLE,
                "axes.titleweight": "bold",
                "axes.labelsize": FONT_SIZE_LABEL,
                "axes.labelcolor": "#222222",
                "axes.prop_cycle": __import__("matplotlib").cycler(color=PALETTE),
                # Grid
                "grid.alpha": GRID_ALPHA,
                "grid.linestyle": GRID_LINESTYLE,
                "grid.linewidth": GRID_LINEWIDTH,
                "grid.color": "#cccccc",
                # Lines
                "lines.linewidth": LINE_WIDTH,
                "lines.markersize": MARKER_SIZE,
                # Font
                "font.family": FONT_FAMILY,
                "font.size": FONT_SIZE_BASE,
                # Ticks
                "xtick.labelsize": FONT_SIZE_TICK,
                "ytick.labelsize": FONT_SIZE_TICK,
                "xtick.direction": "out",
                "ytick.direction": "out",
                "xtick.major.width": SPINE_WIDTH,
                "ytick.major.width": SPINE_WIDTH,
                # Legend
                "legend.fontsize": FONT_SIZE_LEGEND,
                "legend.framealpha": 0.9,
                "legend.edgecolor": "#cccccc",
                "legend.borderpad": 0.5,
                # Saving
                "savefig.dpi": DPI_PRINT,
                "savefig.bbox": "tight",
                "savefig.facecolor": "white",
                "savefig.edgecolor": "none",
                # PDF backend
                "pdf.fonttype": 42,  # TrueType fonts in PDF
                "ps.fonttype": 42,
            }
        )
    except ImportError:
        pass  # matplotlib not available; graceful degradation

    try:
        import seaborn as sns

        sns.set_theme(
            style="whitegrid",
            palette=PALETTE,
            font=FONT_FAMILY,
            font_scale=0.9,
            rc={
                "axes.spines.top": False,
                "axes.spines.right": False,
            },
        )
    except ImportError:
        pass  # seaborn not available; graceful degradation
