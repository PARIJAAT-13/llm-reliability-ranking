"""
BasePlotter — abstract base class for all figure generators.

Purpose
-------
Provides a uniform save/export interface so every concrete plotter can
export to PNG, SVG and PDF without duplicating I/O logic.

Responsibilities
----------------
- Apply the publication style before every figure is produced
- Expose ``save(fig, path, fmt)`` for single-format export
- Expose ``save_all(fig, path)`` for exporting every supported format
- Close figures deterministically to avoid memory leaks in batch runs

Usage example
-------------
>>> from llm_reliability.visualization.plotter import BasePlotter
>>> class MyPlotter(BasePlotter):
...     def plot(self, data):
...         fig, ax = self._new_figure()
...         ax.bar(range(len(data)), data)
...         return fig

Design notes
------------
Concrete plotters must implement ``plot()``.  ``BasePlotter`` does not
enforce a specific signature because different charts accept different
inputs; subclasses document their own parameters.
"""

from __future__ import annotations

import abc
import pathlib
from typing import Any

from llm_reliability.visualization.styles import (DPI_PRINT,
                                                  FIG_HEIGHT_DEFAULT,
                                                  FIG_WIDTH_SINGLE,
                                                  apply_publication_style)


class BasePlotter(abc.ABC):
    """Abstract base class for every visualization in the package.

    Parameters
    ----------
    figsize : tuple[float, float], optional
        Width × height in inches.  Defaults to ``(FIG_WIDTH_SINGLE, FIG_HEIGHT_DEFAULT)``.
    dpi : int, optional
        Dots-per-inch for rasterisation.  Defaults to ``DPI_PRINT``.
    style_applied : bool
        Whether ``apply_publication_style()`` should be called on
        construction.  Set to False when the caller manages the style.
    """

    #: Formats supported by ``save_all``.
    SUPPORTED_FORMATS: tuple[str, ...] = ("png", "svg", "pdf")

    def __init__(
        self,
        figsize: tuple[float, float] | None = None,
        dpi: int = DPI_PRINT,
        style_applied: bool = True,
    ) -> None:
        self._figsize = figsize or (FIG_WIDTH_SINGLE, FIG_HEIGHT_DEFAULT)
        self._dpi = dpi

        if style_applied:
            apply_publication_style()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def plot(self, *args: Any, **kwargs: Any) -> Any:
        """Generate and return a matplotlib Figure.

        Subclasses define their own parameter signatures.
        """

    # ------------------------------------------------------------------
    # Figure helpers
    # ------------------------------------------------------------------

    def _new_figure(
        self,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[Any, Any]:
        """Create a fresh Figure/Axes pair.

        Parameters
        ----------
        figsize : tuple[float, float], optional
            Override the plotter-level figure size.

        Returns
        -------
        tuple[Figure, Axes]
            Matplotlib figure and primary axes.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize or self._figsize)
        return fig, ax

    def _new_figure_grid(
        self,
        nrows: int,
        ncols: int,
        figsize: tuple[float, float] | None = None,
        **subplot_kwargs: Any,
    ) -> tuple[Any, Any]:
        """Create a multi-panel Figure.

        Parameters
        ----------
        nrows : int
            Number of rows.
        ncols : int
            Number of columns.
        figsize : tuple[float, float], optional
            Override figure size.
        **subplot_kwargs
            Forwarded to ``plt.subplots``.

        Returns
        -------
        tuple[Figure, np.ndarray]
            Figure and array of Axes.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize or self._figsize,
            **subplot_kwargs,
        )
        return fig, axes

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def save(
        self,
        fig: Any,
        path: str | pathlib.Path,
        fmt: str = "png",
        close: bool = True,
    ) -> pathlib.Path:
        """Save a figure to *path* in *fmt* format.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to save.
        path : str | Path
            Destination path (extension is appended if absent).
        fmt : str
            Output format: ``"png"``, ``"svg"``, or ``"pdf"``.
        close : bool
            Whether to close the figure after saving (recommended for
            batch runs to avoid memory leaks).

        Returns
        -------
        pathlib.Path
            Resolved path of the saved file.

        Raises
        ------
        ValueError
            If *fmt* is not in ``SUPPORTED_FORMATS``.
        """
        fmt = fmt.lower().lstrip(".")
        if fmt not in self.SUPPORTED_FORMATS:
            msg = f"Unsupported format '{fmt}'. Choose from {self.SUPPORTED_FORMATS}."
            raise ValueError(msg)

        dest = pathlib.Path(path)
        if dest.suffix.lstrip(".").lower() != fmt:
            dest = dest.with_suffix(f".{fmt}")

        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, format=fmt, dpi=self._dpi, bbox_inches="tight")

        if close:
            import matplotlib.pyplot as plt

            plt.close(fig)

        return dest

    def save_all(
        self,
        fig: Any,
        stem: str | pathlib.Path,
    ) -> dict[str, pathlib.Path]:
        """Save a figure in all supported formats.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to save.
        stem : str | Path
            File path without extension.  One file per format is created.

        Returns
        -------
        dict[str, pathlib.Path]
            Mapping of format name → saved path.
        """
        stem = pathlib.Path(stem)
        paths: dict[str, pathlib.Path] = {}
        for fmt in self.SUPPORTED_FORMATS:
            # Only close on last format
            close = fmt == self.SUPPORTED_FORMATS[-1]
            paths[fmt] = self.save(fig, stem, fmt=fmt, close=close)
        return paths

    # ------------------------------------------------------------------
    # Utility: close all open figures
    # ------------------------------------------------------------------

    @staticmethod
    def close_all() -> None:
        """Close every open matplotlib figure (useful after batch runs)."""
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except ImportError:
            pass
