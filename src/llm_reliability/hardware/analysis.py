from __future__ import annotations

import logging
import statistics
from typing import Any

from llm_reliability.records.execution import ExecutionRecord
from llm_reliability.records.metric import MetricRecord

logger = logging.getLogger(__name__)


class HardwareAnalysis:
    @staticmethod
    def reliability_by_ram(
        metrics: list[MetricRecord],
        executions: list[ExecutionRecord],
    ) -> list[dict[str, Any]]:
        hw = _hardware_lookup(executions)
        results: list[dict[str, Any]] = []
        ram_buckets: dict[str, list[float]] = {}
        for m in metrics:
            info = hw.get(m.agent, {})
            ram_gb = info.get("ram_total_gb", 0)
            bucket = _ram_bucket(ram_gb) if ram_gb else "unknown"
            ram_buckets.setdefault(bucket, []).append(m.composite_reliability)
        for bucket, scores in sorted(ram_buckets.items()):
            results.append(
                {
                    "ram_bucket": bucket,
                    "ram_range_gb": _ram_bucket_range(bucket),
                    "model_count": len(scores),
                    "mean_reliability": _safe_mean(scores),
                    "std_reliability": _safe_std(scores),
                    "min_reliability": min(scores) if scores else 0.0,
                    "max_reliability": max(scores) if scores else 0.0,
                }
            )
        return results

    @staticmethod
    def reliability_by_vram(
        metrics: list[MetricRecord],
        executions: list[ExecutionRecord],
    ) -> list[dict[str, Any]]:
        hw = _hardware_lookup(executions)
        results: list[dict[str, Any]] = []
        vram_buckets: dict[str, list[float]] = {}
        for m in metrics:
            info = hw.get(m.agent, {})
            vram_gb = info.get("vram_total_gb", 0)
            bucket = _vram_bucket(vram_gb) if vram_gb else "no-gpu"
            vram_buckets.setdefault(bucket, []).append(m.composite_reliability)
        for bucket, scores in sorted(vram_buckets.items()):
            results.append(
                {
                    "vram_bucket": bucket,
                    "vram_range_gb": _vram_bucket_range(bucket),
                    "model_count": len(scores),
                    "mean_reliability": _safe_mean(scores),
                    "std_reliability": _safe_std(scores),
                    "min_reliability": min(scores) if scores else 0.0,
                    "max_reliability": max(scores) if scores else 0.0,
                }
            )
        return results

    @staticmethod
    def latency_by_ram(
        executions: list[ExecutionRecord],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        ram_buckets: dict[str, list[float]] = {}
        for e in executions:
            meta = dict(e.environment_metadata or {})
            ram_gb = meta.get("ram_total_gb", 0)
            bucket = _ram_bucket(ram_gb) if ram_gb else "unknown"
            ram_buckets.setdefault(bucket, []).append(e.runtime_seconds)
        for bucket, latencies in sorted(ram_buckets.items()):
            results.append(
                {
                    "ram_bucket": bucket,
                    "ram_range_gb": _ram_bucket_range(bucket),
                    "execution_count": len(latencies),
                    "mean_latency_seconds": _safe_mean(latencies),
                    "std_latency_seconds": _safe_std(latencies),
                    "median_latency_seconds": _safe_median(latencies),
                }
            )
        return results

    @staticmethod
    def failure_rate_by_memory(
        executions: list[ExecutionRecord],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        buckets: dict[str, list[bool]] = {}
        for e in executions:
            meta = dict(e.environment_metadata or {})
            ram_gb = meta.get("ram_total_gb", 0)
            bucket = _ram_bucket(ram_gb) if ram_gb else "unknown"
            buckets.setdefault(bucket, []).append(e.status != "success")
        for bucket, failures in sorted(buckets.items()):
            total = len(failures)
            failed = sum(failures)
            results.append(
                {
                    "ram_bucket": bucket,
                    "ram_range_gb": _ram_bucket_range(bucket),
                    "total_executions": total,
                    "failed_executions": failed,
                    "failure_rate": round(failed / total, 4) if total > 0 else 0.0,
                }
            )
        return results

    @staticmethod
    def success_rate_by_hardware(
        metrics: list[MetricRecord],
        executions: list[ExecutionRecord],
    ) -> list[dict[str, Any]]:
        hw = _hardware_lookup(executions)
        results: list[dict[str, Any]] = []
        hw_groups: dict[str, list[float]] = {}
        for m in metrics:
            info = hw.get(m.agent, {})
            profile = info.get("hardware_profile", "unknown")
            hw_groups.setdefault(profile, []).append(m.success_rate)
        for profile, rates in sorted(hw_groups.items()):
            results.append(
                {
                    "hardware_profile": profile,
                    "model_count": len(rates),
                    "mean_success_rate": _safe_mean(rates),
                    "std_success_rate": _safe_std(rates),
                }
            )
        return results

    @staticmethod
    def memory_by_model(
        executions: list[ExecutionRecord],
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        model_data: dict[str, dict[str, Any]] = {}
        for e in executions:
            model = e.agent or "unknown"
            meta = dict(e.environment_metadata or {})
            ram = meta.get("ram_total_gb", 0)
            vram = meta.get("vram_total_gb", 0)
            if model not in model_data:
                model_data[model] = {
                    "model": model,
                    "ram_gb_list": [],
                    "vram_gb_list": [],
                    "execution_count": 0,
                }
            model_data[model]["ram_gb_list"].append(ram)
            model_data[model]["vram_gb_list"].append(vram)
            model_data[model]["execution_count"] += 1
        for model, data in sorted(model_data.items()):
            output.append(
                {
                    "model": model,
                    "mean_ram_gb": _safe_mean(data["ram_gb_list"]),
                    "mean_vram_gb": _safe_mean(data["vram_gb_list"]),
                    "execution_count": data["execution_count"],
                }
            )
        return output

    @staticmethod
    def model_ranking_by_hardware(
        metrics: list[MetricRecord],
        executions: list[ExecutionRecord],
    ) -> dict[str, list[dict[str, Any]]]:
        hw = _hardware_lookup(executions)
        rankings: dict[str, list[dict[str, Any]]] = {}
        hw_groups: dict[str, list[dict[str, Any]]] = {}
        for m in metrics:
            info = hw.get(m.agent, {})
            profile = info.get("hardware_profile", "unknown")
            hw_groups.setdefault(profile, []).append(
                {
                    "model": m.agent,
                    "benchmark": m.benchmark,
                    "success_rate": m.success_rate,
                    "composite_reliability": m.composite_reliability,
                }
            )
        for profile, models in hw_groups.items():
            ranked = sorted(models, key=lambda x: x["composite_reliability"], reverse=True)
            for i, entry in enumerate(ranked):
                entry["rank"] = i + 1
            rankings[profile] = ranked
        return rankings


def _hardware_lookup(
    executions: list[ExecutionRecord],
) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for e in executions:
        meta = dict(e.environment_metadata or {})
        agent = e.agent
        if agent not in lookup:
            lookup[agent] = {
                "ram_total_gb": meta.get("ram_total_gb", 0),
                "vram_total_gb": meta.get("vram_total_gb", 0),
                "hardware_profile": meta.get("hardware_profile", "unknown"),
            }
        else:
            existing = lookup[agent]
            existing["ram_total_gb"] = max(existing["ram_total_gb"], meta.get("ram_total_gb", 0))
            existing["vram_total_gb"] = max(existing["vram_total_gb"], meta.get("vram_total_gb", 0))
    return lookup


def _ram_bucket(ram_gb: float) -> str:
    if ram_gb < 8:
        return "0-8GB"
    if ram_gb < 16:
        return "8-16GB"
    if ram_gb < 32:
        return "16-32GB"
    if ram_gb < 64:
        return "32-64GB"
    return "64GB+"


def _ram_bucket_range(bucket: str) -> str:
    ranges = {
        "0-8GB": "0–8 GB",
        "8-16GB": "8–16 GB",
        "16-32GB": "16–32 GB",
        "32-64GB": "32–64 GB",
        "64GB+": "64+ GB",
        "unknown": "Unknown",
    }
    return ranges.get(bucket, bucket)


def _vram_bucket(vram_gb: float) -> str:
    if vram_gb <= 0:
        return "no-gpu"
    if vram_gb < 6:
        return "0-6GB"
    if vram_gb < 12:
        return "6-12GB"
    if vram_gb < 24:
        return "12-24GB"
    return "24GB+"


def _vram_bucket_range(bucket: str) -> str:
    ranges = {
        "no-gpu": "No GPU",
        "0-6GB": "0–6 GB",
        "6-12GB": "6–12 GB",
        "12-24GB": "12–24 GB",
        "24GB+": "24+ GB",
    }
    return ranges.get(bucket, bucket)


def _safe_mean(values: list[float]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return round(statistics.stdev(values), 4) if len(values) > 1 else 0.0


def _safe_median(values: list[float]) -> float:
    return round(statistics.median(values), 4) if values else 0.0
