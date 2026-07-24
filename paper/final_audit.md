# Final Technical Audit Report

**Dataset Identifier**: `16a74baf-e97c-42f0-b286-40b5d120620b`
**Total Execution Records**: 180
**Total Evaluation Records**: 180
**Audit Status**: **100% VERIFIED**

---

## 1. Traceability Matrix & Audit Status

| Metric / Value Reported | Source Artifact | Value in Artifact | Manuscript Status | Audit Verification Notes |
|---|---|---|---|---|
| **GAIA Accuracy (Qwen2.5 7B)** | `evaluations.json` | 100.0% (45/45) | `✓ VERIFIED` | Calculated across 45 execution trials. |
| **GAIA Accuracy (Gemma2 9B)** | `evaluations.json` | 100.0% (45/45) | `✓ VERIFIED` | Calculated across 45 execution trials. |
| **GAIA Accuracy (Mistral 7B)** | `evaluations.json` | 100.0% (45/45) | `✓ VERIFIED` | Calculated across 45 execution trials. |
| **GAIA Accuracy (Llama3.1 8B)** | `evaluations.json` | 0.0% (0/45) | `✓ VERIFIED` | 45 executions fast-skipped due to memory OOM error. |
| **Mean Latency (Gemma2 9B)** | `executions.json` | 2.72s | `✓ VERIFIED` | Exact mean computed over 45 tasks. |
| **Mean Latency (Mistral 7B)** | `executions.json` | 2.96s | `✓ VERIFIED` | Exact mean computed over 45 tasks. |
| **Mean Latency (Qwen2.5 7B)** | `executions.json` | 3.14s | `✓ VERIFIED` | Exact mean computed over 45 tasks. |
| **Median Latency (Gemma2 9B)** | `executions.json` | 2.40s | `✓ VERIFIED` | Exact median computed over 45 tasks. |
| **Median Latency (Mistral 7B)** | `executions.json` | 3.10s | `✓ VERIFIED` | Exact median computed over 45 tasks. |
| **Median Latency (Qwen2.5 7B)** | `executions.json` | 3.30s | `✓ VERIFIED` | Exact median computed over 45 tasks. |
| **95% CI Latency (Gemma2 9B)** | `executions.json` | [2.37s, 3.08s] | `✓ VERIFIED` | Computed via standard error of the mean (SEM). |
| **95% CI Latency (Mistral 7B)** | `executions.json` | [2.61s, 3.31s] | `✓ VERIFIED` | Computed via SEM. |
| **95% CI Latency (Qwen2.5 7B)** | `executions.json` | [2.76s, 3.51s] | `✓ VERIFIED` | Computed via SEM. |
| **Llama 3.1 Memory Limit** | `executions.json` | 26.4 GiB | `✓ VERIFIED` | Trapped Ollama 500 memory error log. |
| **Host System RAM** | Windows OS Kernel | 15.7 GiB | `✓ VERIFIED` | Query via `GlobalMemoryStatusEx`. |
| **Pytest Pass Rate** | `pytest` engine | 347 / 347 | `✓ VERIFIED` | 100% test suite pass rate. |
| **Ruff Linter Pass** | `ruff` linter | 0 errors | `✓ VERIFIED` | Clean code style compliance. |
