from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_reliability.configs.config import Configuration
from llm_reliability.records.evaluation import EvaluationRecord
from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.records.metric import MetricRecord


class ExperimentDatabase:
    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._create_tables()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def connected(self) -> bool:
        return self._conn is not None

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._conn

    def _create_tables(self) -> None:
        conn = self._ensure_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
            );

            CREATE TABLE IF NOT EXISTS executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                configuration_hash TEXT NOT NULL,
                seed INTEGER NOT NULL,
                benchmark TEXT NOT NULL,
                agent TEXT NOT NULL,
                task_id TEXT NOT NULL,
                run_index INTEGER NOT NULL,
                runtime_seconds REAL NOT NULL,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                agent_output_json TEXT,
                software_versions_json TEXT,
                environment_metadata_json TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            );

            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                execution_hash TEXT NOT NULL,
                configuration_hash TEXT NOT NULL,
                seed INTEGER NOT NULL,
                benchmark TEXT NOT NULL,
                agent TEXT NOT NULL,
                task_id TEXT NOT NULL,
                run_index INTEGER NOT NULL,
                success INTEGER NOT NULL,
                score REAL NOT NULL,
                metrics_json TEXT,
                evaluated_at TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            );

            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                benchmark TEXT NOT NULL,
                agent TEXT NOT NULL,
                task_id TEXT,
                evaluation_count INTEGER NOT NULL,
                success_rate REAL NOT NULL,
                repeated_run_consistency REAL NOT NULL,
                perturbation_robustness REAL,
                fault_tolerance REAL,
                isr_output REAL,
                isr_behavior REAL,
                isr_composite_val REAL,
                composite_reliability REAL NOT NULL,
                computed_at TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            );

            CREATE INDEX IF NOT EXISTS idx_exec_benchmark ON executions(benchmark);
            CREATE INDEX IF NOT EXISTS idx_exec_agent ON executions(agent);
            CREATE INDEX IF NOT EXISTS idx_eval_benchmark ON evaluations(benchmark);
            CREATE INDEX IF NOT EXISTS idx_eval_agent ON evaluations(agent);
            CREATE INDEX IF NOT EXISTS idx_metrics_benchmark ON metrics(benchmark);
            CREATE INDEX IF NOT EXISTS idx_metrics_agent ON metrics(agent);
        """)

    # ------------------------------------------------------------------
    # Experiment CRUD
    # ------------------------------------------------------------------

    def save_experiment(
        self,
        experiment_id: str,
        name: str,
        config: Configuration | None = None,
    ) -> None:
        conn = self._ensure_conn()
        now = datetime.now(timezone.utc).isoformat()
        config_json = json.dumps(config.model_dump() if config else {})
        conn.execute(
            "INSERT OR REPLACE INTO experiments (id, name, config_json, created_at, status) "
            "VALUES (?, ?, ?, ?, ?)",
            (experiment_id, name, config_json, now, "pending"),
        )
        conn.commit()

    def update_experiment_status(self, experiment_id: str, status: str) -> None:
        conn = self._ensure_conn()
        conn.execute("UPDATE experiments SET status = ? WHERE id = ?", (status, experiment_id))
        conn.commit()

    def list_experiments(self) -> list[dict[str, Any]]:
        conn = self._ensure_conn()
        rows = conn.execute(
            "SELECT id, name, created_at, status FROM experiments ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        conn = self._ensure_conn()
        row = conn.execute(
            "SELECT id, name, config_json, created_at, status FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Execution persistence
    # ------------------------------------------------------------------

    def save_execution(
        self,
        experiment_id: str,
        record: ExecutionRecord,
    ) -> int:
        conn = self._ensure_conn()
        cur = conn.execute(
            "INSERT INTO executions "
            "(experiment_id, configuration_hash, seed, benchmark, agent, task_id, "
            "run_index, runtime_seconds, timestamp, status, error, "
            "agent_output_json, software_versions_json, environment_metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                experiment_id,
                record.configuration_hash,
                record.seed,
                record.benchmark,
                record.agent,
                record.task_id,
                record.run_index,
                record.runtime_seconds,
                record.timestamp,
                record.status,
                record.error,
                (
                    json.dumps({"value": record.agent_output})
                    if record.agent_output is not None
                    else None
                ),
                json.dumps(record.software_versions),
                json.dumps(record.environment_metadata),
            ),
        )
        conn.commit()
        return cur.lastrowid or 0

    def save_executions(self, experiment_id: str, records: list[ExecutionRecord]) -> None:
        for r in records:
            self.save_execution(experiment_id, r)

    def query_executions(
        self,
        experiment_id: str | None = None,
        benchmark: str | None = None,
        agent: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        conn = self._ensure_conn()
        conditions = []
        params: list[Any] = []
        if experiment_id:
            conditions.append("experiment_id = ?")
            params.append(experiment_id)
        if benchmark:
            conditions.append("benchmark = ?")
            params.append(benchmark)
        if agent:
            conditions.append("agent = ?")
            params.append(agent)
        if status:
            conditions.append("status = ?")
            params.append(status)
        where = " AND ".join(conditions) if conditions else "1=1"
        rows = conn.execute(
            f"SELECT * FROM executions WHERE {where} ORDER BY timestamp",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Evaluation persistence
    # ------------------------------------------------------------------

    def save_evaluation(
        self,
        experiment_id: str,
        record: EvaluationRecord,
    ) -> int:
        conn = self._ensure_conn()
        cur = conn.execute(
            "INSERT INTO evaluations "
            "(experiment_id, execution_hash, configuration_hash, seed, benchmark, agent, "
            "task_id, run_index, success, score, metrics_json, evaluated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                experiment_id,
                record.execution_hash,
                record.configuration_hash,
                record.seed,
                record.benchmark,
                record.agent,
                record.task_id,
                record.run_index,
                1 if record.success else 0,
                record.score,
                json.dumps(record.metrics) if record.metrics else None,
                record.evaluated_at,
            ),
        )
        conn.commit()
        return cur.lastrowid or 0

    def save_evaluations(self, experiment_id: str, records: list[EvaluationRecord]) -> None:
        for r in records:
            self.save_evaluation(experiment_id, r)

    def query_evaluations(
        self,
        experiment_id: str | None = None,
        benchmark: str | None = None,
        agent: str | None = None,
        successful: bool | None = None,
    ) -> list[dict[str, Any]]:
        conn = self._ensure_conn()
        conditions = []
        params: list[Any] = []
        if experiment_id:
            conditions.append("experiment_id = ?")
            params.append(experiment_id)
        if benchmark:
            conditions.append("benchmark = ?")
            params.append(benchmark)
        if agent:
            conditions.append("agent = ?")
            params.append(agent)
        if successful is not None:
            conditions.append("success = ?")
            params.append(1 if successful else 0)
        where = " AND ".join(conditions) if conditions else "1=1"
        rows = conn.execute(
            f"SELECT * FROM evaluations WHERE {where} ORDER BY evaluated_at",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Metric persistence
    # ------------------------------------------------------------------

    def save_metric(self, experiment_id: str, record: MetricRecord) -> int:
        conn = self._ensure_conn()
        cur = conn.execute(
            "INSERT INTO metrics "
            "(experiment_id, benchmark, agent, task_id, evaluation_count, success_rate, "
            "repeated_run_consistency, perturbation_robustness, fault_tolerance, "
            "isr_output, isr_behavior, isr_composite_val, composite_reliability, computed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                experiment_id,
                record.benchmark,
                record.agent,
                record.task_id,
                record.evaluation_count,
                record.success_rate,
                record.repeated_run_consistency,
                record.perturbation_robustness,
                record.fault_tolerance,
                record.isr_output,
                record.isr_behavior,
                record.isr_composite_val,
                record.composite_reliability,
                record.computed_at,
            ),
        )
        conn.commit()
        return cur.lastrowid or 0

    def save_metrics(self, experiment_id: str, records: list[MetricRecord]) -> None:
        for r in records:
            self.save_metric(experiment_id, r)

    def query_metrics(
        self,
        experiment_id: str | None = None,
        benchmark: str | None = None,
        agent: str | None = None,
    ) -> list[dict[str, Any]]:
        conn = self._ensure_conn()
        conditions = []
        params: list[Any] = []
        if experiment_id:
            conditions.append("experiment_id = ?")
            params.append(experiment_id)
        if benchmark:
            conditions.append("benchmark = ?")
            params.append(benchmark)
        if agent:
            conditions.append("agent = ?")
            params.append(agent)
        where = " AND ".join(conditions) if conditions else "1=1"
        rows = conn.execute(
            f"SELECT * FROM metrics WHERE {where} ORDER BY composite_reliability DESC",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Cross-experiment comparison
    # ------------------------------------------------------------------

    def compare_agents_across_experiments(
        self,
        experiment_ids: list[str],
    ) -> list[dict[str, Any]]:
        conn = self._ensure_conn()
        placeholders = ",".join("?" for _ in experiment_ids)
        rows = conn.execute(
            f"SELECT m.experiment_id, e.name AS experiment_name, "
            f"m.benchmark, m.agent, m.success_rate, m.composite_reliability, "
            f"m.computed_at "
            f"FROM metrics m "
            f"JOIN experiments e ON m.experiment_id = e.id "
            f"WHERE m.experiment_id IN ({placeholders}) "
            f"ORDER BY m.benchmark, m.agent, e.created_at",
            experiment_ids,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_ranking(
        self,
        experiment_id: str,
        benchmark: str | None = None,
    ) -> list[dict[str, Any]]:
        conn = self._ensure_conn()
        conditions = ["experiment_id = ?"]
        params: list[Any] = [experiment_id]
        if benchmark:
            conditions.append("benchmark = ?")
            params.append(benchmark)
        where = " AND ".join(conditions)
        rows = conn.execute(
            f"SELECT agent, benchmark, success_rate, composite_reliability "
            f"FROM metrics WHERE {where} "
            f"ORDER BY composite_reliability DESC",
            params,
        ).fetchall()
        return [dict(r) for r in rows]
