"""Export experiment records (ExecutionRecord, EvaluationRecord, etc.) to CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.records.metric import MetricRecord
from llm_reliability.records.ranking import RankingRecord


class RecordExporter:
    """Export experiment record lists to CSV files.

    All methods are class methods for stateless usage.
    """

    @classmethod
    def _records_to_dataframe(cls, records: list[Any]) -> Any:
        """Convert a list of SerializableModel records to a pandas DataFrame."""
        import pandas as pd
        rows = [r.canonical_dict() for r in records]
        return pd.DataFrame(rows)

    @classmethod
    def export_executions(
        cls,
        records: list[ExecutionRecord],
        path: str | Path,
    ) -> Path:
        """Export ExecutionRecords to CSV.

        Parameters
        ----------
        records : list[ExecutionRecord]
        path : str | Path
            Destination path (``.csv`` suffix added if missing).

        Returns
        -------
        Path
        """
        dest = Path(path)
        if dest.suffix.lower() != ".csv":
            dest = dest.with_suffix(".csv")
        dest.parent.mkdir(parents=True, exist_ok=True)
        df = cls._records_to_dataframe(records)
        df.to_csv(dest, index=False)
        return dest

    @classmethod
    def export_evaluations(
        cls,
        records: list[EvaluationRecord],
        path: str | Path,
    ) -> Path:
        """Export EvaluationRecords to CSV."""
        dest = Path(path).with_suffix(".csv")
        dest.parent.mkdir(parents=True, exist_ok=True)
        df = cls._records_to_dataframe(records)
        df.to_csv(dest, index=False)
        return dest

    @classmethod
    def export_metrics(
        cls,
        records: list[MetricRecord],
        path: str | Path,
    ) -> Path:
        """Export MetricRecords to CSV."""
        dest = Path(path).with_suffix(".csv")
        dest.parent.mkdir(parents=True, exist_ok=True)
        df = cls._records_to_dataframe(records)
        df.to_csv(dest, index=False)
        return dest

    @classmethod
    def export_rankings(
        cls,
        records: list[RankingRecord],
        path: str | Path,
    ) -> Path:
        """Export RankingRecords to CSV."""
        dest = Path(path).with_suffix(".csv")
        dest.parent.mkdir(parents=True, exist_ok=True)
        df = cls._records_to_dataframe(records)
        df.to_csv(dest, index=False)
        return dest

    @classmethod
    def export_all(
        cls,
        executions: list[ExecutionRecord] | None = None,
        evaluations: list[EvaluationRecord] | None = None,
        metrics: list[MetricRecord] | None = None,
        rankings: list[RankingRecord] | None = None,
        output_dir: str | Path = "exports",
    ) -> dict[str, Path]:
        """Export all provided record types to CSV in a single directory.

        Parameters
        ----------
        executions : list[ExecutionRecord] | None
        evaluations : list[EvaluationRecord] | None
        metrics : list[MetricRecord] | None
        rankings : list[RankingRecord] | None
        output_dir : str | Path
            Output directory for CSV files.

        Returns
        -------
        dict[str, Path]
            Mapping of record type name → file path.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        result: dict[str, Path] = {}

        if executions:
            result["executions"] = cls.export_executions(executions, out / "executions.csv")
        if evaluations:
            result["evaluations"] = cls.export_evaluations(evaluations, out / "evaluations.csv")
        if metrics:
            result["metrics"] = cls.export_metrics(metrics, out / "metrics.csv")
        if rankings:
            result["rankings"] = cls.export_rankings(rankings, out / "rankings.csv")

        return result
