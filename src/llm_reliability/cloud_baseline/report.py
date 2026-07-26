from __future__ import annotations

from llm_reliability.cloud_baseline.models import (CloudBaselineComparison,
                                                   CloudBaselineResult,
                                                   CloudBaselineSummary)


class CloudBaselineReportGenerator:
    @staticmethod
    def summary_table(results: list[CloudBaselineResult]) -> str:
        summaries = CloudBaselineSummary.compute_all(results)
        if not summaries:
            return "No results to report."

        lines = [
            "# Cloud Baseline Comparison",
            "",
            "| Provider | Model | Benchmark | Success Rate | Avg Latency (ms) | P95 Latency | Total Cost ($) | Cost/Success ($) |",
            "|----------|-------|-----------|-------------|-----------------|-------------|---------------|------------------|",
        ]
        for s in summaries:
            lines.append(
                f"| {s.provider} | {s.model} | {s.benchmark} "
                f"| {s.success_rate:.1%} | {s.avg_latency_ms:.1f} "
                f"| {s.p95_latency_ms:.1f} | ${float(s.total_cost_usd):.4f} "
                f"| ${float(s.cost_per_success_usd):.4f} |"
            )

        lines.append("")
        comp = CloudBaselineComparison.compute(summaries)
        if comp is not None and comp.best_provider:
            lines.append(
                f"**Best accuracy**: {comp.best_provider}/{comp.best_model} "
                f"({comp.best_score:.1%})"
            )
        if comp is not None and comp.most_efficient_provider:
            lines.append(
                f"**Most cost-efficient**: {comp.most_efficient_provider}/"
                f"{comp.most_efficient_model} "
                f"(${float(comp.lowest_cost_per_success):.4f}/success)"
            )

        return "\n".join(lines)

    @staticmethod
    def csv_report(results: list[CloudBaselineResult]) -> str:
        lines = [
            "provider,model,benchmark,task_id,success,score,cost_usd,latency_ms,"
            "tokens_input,tokens_output,runtime_seconds"
        ]
        for r in results:
            lines.append(
                f"{r.provider},{r.model},{r.benchmark},{r.task_id},"
                f"{r.success},{r.score},{r.cost_usd},{r.latency_ms:.1f},"
                f"{r.tokens_input},{r.tokens_output},{r.runtime_seconds:.2f}"
            )
        return "\n".join(lines)

    @staticmethod
    def detailed_report(results: list[CloudBaselineResult]) -> str:
        parts = [CloudBaselineReportGenerator.summary_table(results)]
        parts.append("")
        for r in results:
            if r.error:
                status = f"ERROR: {r.error}"
            else:
                cost_str = f"${float(r.cost_usd):.6f}"
                status = (
                    f"score={r.score:.2f} cost={cost_str} "
                    f"latency={r.latency_ms:.1f}ms tokens=({r.tokens_input}+{r.tokens_output})"
                )
            parts.append(f"- `{r.provider}/{r.model}` on `{r.benchmark}#{r.task_id}`: {status}")
        return "\n".join(parts)
