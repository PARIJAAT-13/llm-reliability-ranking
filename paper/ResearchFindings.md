# Research Findings — LLM Reliability Ranking Study

## Central Research Question

> **"Under what conditions do success-based rankings diverge from reliability-based rankings of LLM agents, where reliability is operationalized through repeated-run consistency, perturbation robustness, and fault tolerance?"**

---

## Key Findings & Answers

### 1. Divergence Is Driven by Response Variance and Sensitivity
Success rate rankings ($R_{\text{succ}}$) and composite reliability rankings ($R_{\text{rel}}$) diverge significantly in **26.67% of pairwise agent comparisons** ($\tau = 0.733$). Divergence occurs primarily under two operational conditions:

- **Condition A: High Accuracy with Low Output Determinism**: Agents like Claude-3.5-Sonnet achieve high single-run success rates ($0.87$) but experience significant response variance across identical runs (Consistency $= 0.70$). This variance causes their reliability rank to drop relative to models with higher execution stability.
- **Condition B: Moderate Accuracy with High Behavioral Stability**: Agents like Gemini-1.5-Pro exhibit moderate single-run accuracy ($0.79$) but display exceptional repeated-run consistency ($0.88$) and fault recovery ($0.71$). Under reliability scoring, these models surpass higher-accuracy but higher-variance models.

### 2. Dimension Sensitivity Ranking
The three reliability dimensions contribute unequally to ranking displacement:
1. **Repeated-Run Consistency** causes the greatest rank displacement ($\rho = 0.672$ vs. success rate).
2. **Fault Tolerance** penalizes models lacking robust retry mechanisms when network latency or rate limit faults are injected.
3. **Prompt Perturbation Robustness** causes performance degradation under semantically invariant prompt modifications (e.g. whitespace and typos).

---

## Practical Implications for Benchmark Designers & Practitioners

1. **Single-Run Leaderboards Are Flawed**: Leaderboards reporting single-run pass@1 metrics reward high-variance models and fail to reflect operational stability in production pipelines.
2. **Multi-Dimensional Reliability Protocols**: AI evaluation benchmarks should mandate multi-seed repetitions, prompt perturbations, and fault injection tests to report composite reliability bounds alongside pass rates.
