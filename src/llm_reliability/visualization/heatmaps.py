"""
Heatmap visualizations for correlation matrices.

Purpose
-------
Produce annotated heatmaps of Kendall Tau and Spearman rank correlations
between pairs of ranking types or benchmarks.

Responsibilities
----------------
- Render a pairwise correlation matrix as a colour-coded heatmap
- Annotate every cell with its numerical value
- Produce separate heatmaps for Kendall Tau and Spearman
- Accept either raw matrices or ``CorrelationResult`` dictionaries

Usage example
-------------
>>> from llm_reliability.visualization.heatmaps import HeatmapPlotter
>>> plotter = HeatmapPlotter()
>>> matrix = {"A_vs_B": 0.85, "A_vs_C": 0.42}
>>> fig = plotter.plot_from_dict(matrix, title="Kendall Tau")

How figures are produced
------------------------
``HeatmapPlotter.plot_matrix`` receives a square NumPy array (or Pandas
DataFrame) and uses ``seaborn.heatmap`` with a diverging colourmap centred
at zero.  Annotations are formatted to two decimal places.
"""

from __future__ import annotations

from typing import Any

from llm_reliability.visualization.plotter import BasePlotter
from llm_reliability.visualization.styles import (
    CMAP_DIVERGING,
    FIG_WIDTH_DOUBLE,
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_TITLE,
)


class HeatmapPlotter(BasePlotter):
    """Generate annotated correlation heatmaps.

    Parameters
    ----------
    figsize : tuple[float, float], optional
        Override default figure size.
    """

    def plot(  # type: ignore[override]
        self,
        matrix: Any,
        labels: list[str] | None = None,
        title: str = "Correlation Heatmap",
        vmin: float = -1.0,
        vmax: float = 1.0,
        cmap: str = CMAP_DIVERGING,
        fmt: str = ".2f",
        figsize: tuple[float, float] | None = None,
    ) -> Any:
        """Render a square correlation matrix as an annotated heatmap.

        Parameters
        ----------
        matrix : array-like or pd.DataFrame
            Square matrix of correlation values.
        labels : list[str], optional
            Row/column labels.  Required when *matrix* is a plain array.
        title : str
            Figure title.
        vmin : float
            Minimum value for the colour scale.
        vmax : float
            Maximum value for the colour scale.
        cmap : str
            Matplotlib colourmap name.
        fmt : str
            Number format string for cell annotations.
        figsize : tuple[float, float], optional
            Override figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import numpy as np

        try:
            import pandas as pd
            import seaborn as sns

            _seaborn_available = True
        except ImportError:
            _seaborn_available = False

        if figsize is None:
            n = (
                len(labels)
                if labels
                else (matrix.shape[0] if hasattr(matrix, "shape") else len(matrix))
            )
            side = max(FIG_WIDTH_DOUBLE * 0.5, min(FIG_WIDTH_DOUBLE, n * 0.6))
            figsize = (side, side * 0.85)

        fig, ax = self._new_figure(figsize=figsize)

        try:
            import pandas as pd

            if not isinstance(matrix, pd.DataFrame):
                arr = np.array(matrix, dtype=float)
                df = pd.DataFrame(arr, index=labels, columns=labels)
            else:
                df = matrix
        except ImportError:
            import numpy as np

            arr = np.array(matrix, dtype=float)
            df = arr

        if _seaborn_available:
            import seaborn as sns

            sns.heatmap(
                df,
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                annot=True,
                fmt=fmt,
                linewidths=0.5,
                linecolor="#dddddd",
                square=True,
                annot_kws={"size": FONT_SIZE_ANNOTATION},
                cbar_kws={"shrink": 0.8},
            )
        else:
            im = ax.imshow(df, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
            plt.colorbar(im, ax=ax, shrink=0.8)
            arr_np = np.array(df)
            for i in range(arr_np.shape[0]):
                for j in range(arr_np.shape[1]):
                    ax.text(
                        j,
                        i,
                        f"{arr_np[i, j]:{fmt}}",
                        ha="center",
                        va="center",
                        fontsize=FONT_SIZE_ANNOTATION,
                        color="white" if abs(arr_np[i, j]) > 0.5 else "black",
                    )
            if labels:
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.set_yticklabels(labels, rotation=0)

        ax.set_title(title, fontsize=FONT_SIZE_TITLE, pad=8)
        fig.tight_layout()
        return fig

    def plot_from_dict(
        self,
        correlations: dict[str, Any],
        title: str = "Correlation Heatmap",
    ) -> Any:
        """Build a 2×2 heatmap from a ``{name: CorrelationResult}`` dictionary.

        This is a convenience wrapper for the output of
        ``StatisticalEngine.compute_correlations``.

        Parameters
        ----------
        correlations : dict[str, CorrelationResult]
            Mapping from method name (e.g. ``"spearman"``) to result object.
        title : str
            Figure title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import numpy as np

        methods = list(correlations.keys())
        n = len(methods)
        matrix = np.eye(n)
        for i, m in enumerate(methods):
            result = correlations[m]
            coeff = result.coefficient if hasattr(result, "coefficient") else float(result)
            matrix[i, i] = coeff
            for j in range(i):
                m2 = methods[j]
                r2 = correlations[m2]
                c2 = r2.coefficient if hasattr(r2, "coefficient") else float(r2)
                cross = (coeff + c2) / 2
                matrix[i, j] = cross
                matrix[j, i] = cross

        return self.plot(matrix, labels=methods, title=title)

    def plot_pairwise_matrix(
        self,
        agent_names: list[str],
        score_matrix: Any,
        title: str = "Pairwise Correlation",
        method: str = "spearman",
    ) -> Any:
        """Compute and plot a pairwise correlation matrix across agents.

        Parameters
        ----------
        agent_names : list[str]
            Names of agents (rows/cols of the matrix).
        score_matrix : array-like, shape (n_agents, n_runs)
            Score matrix; correlations are computed across runs.
        title : str
            Plot title.
        method : str
            ``"spearman"`` or ``"pearson"``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import numpy as np

        try:
            import pandas as pd

            df = pd.DataFrame(score_matrix, index=agent_names).T
            corr = df.corr(method=method)
        except ImportError:
            arr = np.array(score_matrix, dtype=float)
            n = arr.shape[0]
            corr_arr = np.eye(n)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        x, y = arr[i], arr[j]
                        if np.std(x) > 0 and np.std(y) > 0:
                            corr_arr[i, j] = float(np.corrcoef(x, y)[0, 1])
            corr = corr_arr

        return self.plot(corr, labels=agent_names, title=f"{title} ({method})")
