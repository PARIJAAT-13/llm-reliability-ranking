# Technical Audit Report: LLM Reliability Ranking Framework

This report presents a technical audit of the repository code (`src/llm_reliability/`), experiment execution artifacts (`results/full_study/ea591e92-faaf-422f-9c44-116b7034f198`), generated figures (`paper/figures/`), and manuscript documents (`paper/*.md`).

---

## 1. Audit Summary & Verification Matrix

| Audit Focus Area | Verification Status | Primary Artifact Source | Findings |
|---|---|---|---|
| **1. Numerical Consistency** | **Major Discrepancy** | `results/full_study/.../executions.json` | Mean latencies reported in text/tables (2.08s–2.41s) reflect an earlier baseline run; actual `runtime_seconds` in the latest JSON are 2.98s–4.10s. |
| **2. Figure Traceability** | **VERIFIED** | `paper/figures/*.png` | All 5 PNG plots match the newest experiment directory. |
| **3. Table Consistency** | **VERIFIED** | `executions.json` & `evaluations.json` | Task counts (5/5 vs 0/5), accuracy (100% vs 0%), and exact-match string outputs match JSON files. |
| **4. Methodology Claims** | **VERIFIED** | `src/llm_reliability/` codebase | All functions (`estimate_model_memory`, `unload_ollama_model`, `normalize_gaia_answer`) exist in code. |
| **5. Architecture Alignment** | **VERIFIED** | `src/llm_reliability/pipeline/` | Component interfaces and data flows match code structure. |
| **6. Claims & Scope** | **Minor Note** | `data/gaia_sample.json` | Manuscript mentions GAIA Levels 1–3, whereas the empirical test suite evaluates 5 GAIA Level 1 tasks. |

---

## 2. Categorized Findings & Evidence

### 2.1 Critical Findings (Severity: High)
*None identified. The codebase compiles cleanly, passes all 347 unit tests (`pytest`), passes static linting (`ruff`), and executes deterministically.*

---

### 2.2 Major Findings (Severity: Medium)

#### Finding M-01: Discrepancy Between Reported Mean Latencies and `executions.json` `runtime_seconds`
- **Location**: `paper/results.md` (Table 1, Table 3, Section 2) & `paper/methodology.md`
- **Description**: The manuscript tables list mean response latencies as:
  - `gemma2:9b`: 2.08s
  - `mistral:7b`: 2.15s
  - `qwen2.5:7b`: 2.41s
- **Empirical Evidence from Latest JSON (`executions.json`)**:
  - `qwen2.5:7b`: Task runtimes = `[1.3s, 1.7s, 4.8s, 4.5s, 2.6s]` $\rightarrow$ **Actual Mean = 2.98s**
  - `gemma2:9b`: Task runtimes = `[1.9s, 3.3s, 2.8s, 3.4s, 4.7s]` $\rightarrow$ **Actual Mean = 3.22s**
  - `mistral:7b`: Task runtimes = `[4.4s, 2.0s, 4.6s, 4.8s, 4.7s]` $\rightarrow$ **Actual Mean = 4.10s**
- **Impact**: The text reports latencies from a prior trial run without system prompt overhead rather than the exact mean `runtime_seconds` from the latest experiment payload (`ea591e92-faaf-422f-9c44-116b7034f198`).
- **Recommended Remediation**: Update Table 1 and Table 3 in `paper/results.md` to reflect exact means (2.98s for Qwen2.5, 3.22s for Gemma2, 4.10s for Mistral).

---

### 2.3 Minor Findings (Severity: Low)

#### Finding L-01: Dataset Level Scope Ambiguity
- **Location**: `paper/methodology.md` (Section 3.4) & `paper/results.md` (Section 1.2)
- **Description**: The methodology mentions supporting GAIA Level 1–3 questions, whereas the sample dataset fixture (`data/gaia_sample.json`) evaluates 5 GAIA Level 1 short-answer questions.
- **Evidence**: `data/gaia_sample.json` metadata contains `"level": 1` for all 5 tasks (`gaia_001` through `gaia_005`).
- **Recommended Remediation**: Clarify in `paper/methodology.md` that the framework architecture supports Level 1–3 schemas, while the current experiment fixture focuses on 5 Level 1 validation tasks.

#### Finding L-02: Model Memory Specification Reporting
- **Location**: `paper/results.md` (Table 2 & Section 4.2)
- **Description**: Text uses "26.4 GiB RAM" interchangeably with "26.4 GB".
- **Evidence**: Ollama API memory logs report `26.4 GiB` binary gigabytes.
- **Recommended Remediation**: Standardize notation to `26.4 GiB` across all tables and manuscript sections.

---

### 2.4 Editorial Findings (Severity: Informational)

#### Finding E-01: Figure Caption Alignment
- **Location**: `paper/results.md` (Section 3)
- **Description**: Figure captions use absolute file links (`file:///c:/.../paper/figures/accuracy.png`) for local rendering in IDE viewers.
- **Recommended Remediation**: When compiling final LaTeX or PDF camera-ready copies, convert relative image paths to standard `\includegraphics{figures/accuracy.pdf}` format.

---

## 3. Code & Implementation Audit Findings

1. **`estimate_model_memory()`**: Verified in `src/llm_reliability/agents/utils/ollama_utils.py` (Line 116).
2. **`unload_ollama_model()`**: Verified in `src/llm_reliability/agents/utils/ollama_utils.py` (Line 193) with parameter `keep_alive: 0`.
3. **`OllamaMemoryError`**: Verified in `src/llm_reliability/exceptions.py` with `is_transient = False`.
4. **`normalize_gaia_answer()`**: Verified in `src/llm_reliability/benchmarks/adapters/gaia_adapter.py`.
5. **System Prompt Decoupling**: Verified in `src/llm_reliability/experiments/experiment_runner.py` and `src/llm_reliability/agents/ollama_agent.py`.
