"""
Export utilities for figures and tables.

Purpose
-------
Provide ``FigureExporter`` and ``TableExporter`` — thin wrappers around
matplotlib and pandas I/O that handle directory creation, format dispatch,
and logging for batch export runs.

Responsibilities
----------------
- ``FigureExporter``: PNG, SVG, PDF from matplotlib Figure objects
- ``TableExporter``: CSV, Excel, JSON, Markdown, LaTeX from DataFrames

Usage example
-------------
>>> from llm_reliability.visualization.export import FigureExporter, TableExporter
>>> FigureExporter.save(fig, "results/figures/ranking", fmt="png")
>>> TableExporter.save_csv(df, "results/tables/metrics.csv")

How exports are produced
------------------------
Both exporters are stateless collections of class methods so they can be
used without instantiation.  Directory creation is handled automatically.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

logger = logging.getLogger(__name__)


class FigureExporter:
    """Export matplotlib figures to raster and vector formats.

    All methods are class methods; no instantiation is required.
    """

    #: Formats this exporter supports.
    SUPPORTED_FORMATS: tuple[str, ...] = ("png", "svg", "pdf")

    @classmethod
    def save(
        cls,
        fig: Any,
        path: str | pathlib.Path,
        fmt: str = "png",
        dpi: int = 300,
        close: bool = True,
    ) -> pathlib.Path:
        """Save a matplotlib Figure to disk.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to save.
        path : str | Path
            Destination path (extension is appended / replaced as needed).
        fmt : str
            One of ``"png"``, ``"svg"``, ``"pdf"``.
        dpi : int
            Resolution for rasterisation (ignored for SVG/PDF).
        close : bool
            Close the figure after saving.

        Returns
        -------
        pathlib.Path
            The resolved output path.

        Raises
        ------
        ValueError
            If *fmt* is unsupported.
        """
        fmt = fmt.lower().lstrip(".")
        if fmt not in cls.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported figure format '{fmt}'. Choose from {cls.SUPPORTED_FORMATS}."
            )

        dest = pathlib.Path(path).with_suffix(f".{fmt}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, format=fmt, dpi=dpi, bbox_inches="tight")
        logger.debug("Figure saved: %s", dest)

        if close:
            try:
                import matplotlib.pyplot as plt

                plt.close(fig)
            except ImportError:
                pass

        return dest

    @classmethod
    def save_all(
        cls,
        fig: Any,
        stem: str | pathlib.Path,
        dpi: int = 300,
    ) -> dict[str, pathlib.Path]:
        """Save a figure in PNG, SVG, and PDF simultaneously.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
        stem : str | Path
            Path without extension.
        dpi : int

        Returns
        -------
        dict[str, pathlib.Path]
            Mapping of format → saved path.
        """
        stem = pathlib.Path(stem)
        paths: dict[str, pathlib.Path] = {}
        for i, fmt in enumerate(cls.SUPPORTED_FORMATS):
            close = i == len(cls.SUPPORTED_FORMATS) - 1
            paths[fmt] = cls.save(fig, stem, fmt=fmt, dpi=dpi, close=close)
        return paths


class TableExporter:
    """Export Pandas DataFrames to multiple tabular formats.

    All methods are class methods; no instantiation is required.
    """

    @classmethod
    def save_csv(
        cls,
        df: Any,
        path: str | pathlib.Path,
        index: bool = False,
        **kwargs: Any,
    ) -> pathlib.Path:
        """Save a DataFrame as CSV.

        Parameters
        ----------
        df : pd.DataFrame
        path : str | Path
        index : bool
        **kwargs
            Forwarded to ``df.to_csv``.

        Returns
        -------
        pathlib.Path
        """
        dest = pathlib.Path(path).with_suffix(".csv")
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=index, **kwargs)
        logger.debug("Table saved (CSV): %s", dest)
        return dest

    @classmethod
    def save_excel(
        cls,
        df: Any,
        path: str | pathlib.Path,
        sheet_name: str = "Results",
        index: bool = False,
        **kwargs: Any,
    ) -> pathlib.Path:
        """Save a DataFrame as Excel (.xlsx).

        Parameters
        ----------
        df : pd.DataFrame
        path : str | Path
        sheet_name : str
        index : bool
        **kwargs
            Forwarded to ``df.to_excel``.

        Returns
        -------
        pathlib.Path

        Raises
        ------
        ImportError
            If ``openpyxl`` is not installed.
        """
        dest = pathlib.Path(path).with_suffix(".xlsx")
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_excel(dest, sheet_name=sheet_name, index=index, **kwargs)
        except ImportError as exc:
            raise ImportError(
                "openpyxl is required for Excel export. Install it with: pip install openpyxl"
            ) from exc
        logger.debug("Table saved (Excel): %s", dest)
        return dest

    @classmethod
    def save_json(
        cls,
        df: Any,
        path: str | pathlib.Path,
        orient: str = "records",
        indent: int = 2,
        **kwargs: Any,
    ) -> pathlib.Path:
        """Save a DataFrame as JSON.

        Parameters
        ----------
        df : pd.DataFrame
        path : str | Path
        orient : str
            JSON orientation string for ``df.to_json``.
        indent : int
        **kwargs

        Returns
        -------
        pathlib.Path
        """
        dest = pathlib.Path(path).with_suffix(".json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(dest, orient=orient, indent=indent, **kwargs)
        logger.debug("Table saved (JSON): %s", dest)
        return dest

    @classmethod
    def save_markdown(
        cls,
        df: Any,
        path: str | pathlib.Path,
        index: bool = False,
        **kwargs: Any,
    ) -> pathlib.Path:
        """Save a DataFrame as Markdown.

        Parameters
        ----------
        df : pd.DataFrame
        path : str | Path
        index : bool
        **kwargs

        Returns
        -------
        pathlib.Path
        """
        dest = pathlib.Path(path).with_suffix(".md")
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            md = df.to_markdown(index=index, **kwargs)
        except Exception:
            md = df.to_string(index=index)
        dest.write_text(md or "", encoding="utf-8")
        logger.debug("Table saved (Markdown): %s", dest)
        return dest

    @classmethod
    def save_latex(
        cls,
        df: Any,
        path: str | pathlib.Path,
        caption: str = "",
        label: str = "tab:results",
        index: bool = False,
        float_format: str = "%.4f",
        **kwargs: Any,
    ) -> pathlib.Path:
        """Save a DataFrame as a LaTeX table.

        Parameters
        ----------
        df : pd.DataFrame
        path : str | Path
        caption : str
        label : str
        index : bool
        float_format : str
        **kwargs

        Returns
        -------
        pathlib.Path
        """
        dest = pathlib.Path(path).with_suffix(".tex")
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            latex = df.to_latex(
                index=index,
                caption=caption,
                label=label,
                float_format=float_format,
                escape=True,
                **kwargs,
            )
        except Exception:
            latex = df.to_string(index=index)
        dest.write_text(latex or "", encoding="utf-8")
        logger.debug("Table saved (LaTeX): %s", dest)
        return dest

    @classmethod
    def save_all(
        cls,
        df: Any,
        stem: str | pathlib.Path,
        caption: str = "",
        label: str = "tab:results",
        skip_excel: bool = False,
    ) -> dict[str, pathlib.Path]:
        """Save a DataFrame in CSV, JSON, Markdown, and LaTeX formats.

        Parameters
        ----------
        df : pd.DataFrame
        stem : str | Path
            Path without extension.
        caption : str
            LaTeX caption.
        label : str
            LaTeX label.
        skip_excel : bool
            Skip Excel export (useful when openpyxl is not installed).

        Returns
        -------
        dict[str, pathlib.Path]
        """
        stem = pathlib.Path(stem)
        paths: dict[str, pathlib.Path] = {
            "csv": cls.save_csv(df, stem),
            "json": cls.save_json(df, stem),
            "markdown": cls.save_markdown(df, stem),
            "latex": cls.save_latex(df, stem, caption=caption, label=label),
        }
        if not skip_excel:
            try:
                paths["excel"] = cls.save_excel(df, stem)
            except ImportError:
                logger.warning("openpyxl not installed; skipping Excel export.")
        return paths
